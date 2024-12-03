import torch
import time
import numpy as np
from torch import nn
from torchvision.utils import make_grid
from torchvision import transforms
from utils import transforms as local_transforms
from base import BaseTrainer, DataPrefetcher
from utils.helpers import colorize_mask
from utils.metrics import eval_metrics, AverageMeter
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def upsample(tensor, size):
    return F.interpolate(tensor, size, mode='bilinear', align_corners=True)


bce_loss = nn.BCELoss(reduction='mean')
CE = torch.nn.BCEWithLogitsLoss()


class Trainer(BaseTrainer):
    def __init__(self, model, loss, resume, config, train_loader, val_loader=None, train_logger=None,
                 prefetch=True):
        super(Trainer, self).__init__(model, loss, resume, config, train_loader, val_loader, train_logger)

        self.wrt_mode, self.wrt_step = 'train_', 0
        self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(self.train_loader.batch_size)))
        if config['trainer']['log_per_iter']: self.log_step = int(self.log_step / self.train_loader.batch_size) + 1

        self.num_classes = self.train_loader.dataset.num_classes
        self.upsample_2 = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_3 = torch.nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            local_transforms.DeNormalize(self.train_loader.MEAN, self.train_loader.STD),
            transforms.ToPILImage()])
        self.viz_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()])

        if self.device == torch.device('cpu'): prefetch = False
        if prefetch:
            self.train_loader = DataPrefetcher(train_loader, device=self.device)
            self.val_loader = DataPrefetcher(val_loader, device=self.device)

        torch.backends.cudnn.benchmark = True

    def _train_epoch(self, epoch):
        global loss
        self.logger.info('\n')

        self.model.train()
        if self.config['arch']['args']['freeze_bn']:
            if isinstance(self.model, torch.nn.DataParallel):
                self.model.module.freeze_bn()
            else:
                self.model.freeze_bn()
        self.wrt_mode = 'train'

        tic = time.time()
        self._reset_metrics()
        tbar = tqdm(self.train_loader, ncols=130)
        for batch_idx, (data, target, edge_gt) in enumerate(tbar):
            self.data_time.update(time.time() - tic)
            data, target, edge_gt = data.to(self.device), target.to(self.device), edge_gt.to(self.device)
            self.lr_scheduler.step(epoch=epoch - 1)

            # LOSS & OPTIMIZE
            self.optimizer.zero_grad()
            output_1, output_2, output_3, output_4, output_5, edge_1 = self.model(data)
            # output_1, output_2, output_3, output_4, output_5 = self.model(data)
            output_1, output_2, output_3, output_4, output_5, edge_1, edge_2, edge_3, edge_4 = self.model(data)
            # output_1, edge_1, edge_2, edge_3, edge_4 = self.model(data)
            # output_1 = self.model(data)
            if self.config['arch']['type'][:3] == 'PSP':
                assert output.size()[2:] == target.size()[1:]
                assert output.size()[0] == self.num_classes
                # loss = self.loss(output.size()[1:], target)
                output = output[0]
            else:
                assert output_1.size()[2:] == target.size()[1:]
                assert output_1.size()[1] == self.num_classes
                loss1 = self.loss(output_1, target)
                loss2 = self.loss(output_2, target)
                loss3 = self.loss(output_3, target)
                loss4 = self.loss(output_4, target)
                loss5 = self.loss(output_5, target)
                loss = loss1 + loss2 + loss3 + loss4 + loss5
                # loss = loss1
            edge_3 = self.upsample_2(edge_3)
            edge_4 = self.upsample_3(edge_4)
            edge_loss_1 = CE(edge_1, edge_gt.unsqueeze(dim=1).float())
            edge_loss_2 = CE(edge_2, edge_gt.unsqueeze(dim=1).float())
            edge_loss_3 = CE(edge_3, edge_gt.unsqueeze(dim=1).float())
            edge_loss_4 = CE(edge_4, edge_gt.unsqueeze(dim=1).float())
            total_loss = loss + edge_loss_1 + edge_loss_2 + edge_loss_3 + edge_loss_4
            # total_loss = loss + edge_loss_1
            # total_loss = loss
            if isinstance(self.loss, torch.nn.DataParallel):
                total_loss = total_loss.mean()
            total_loss.backward()
            self.optimizer.step()
            # self.total_loss.update(loss.item())
            self.total_loss.update(total_loss.item())

            # measure elapsed time
            self.batch_time.update(time.time() - tic)
            tic = time.time()

            # LOGGING & TENSORBOARD
            if batch_idx % self.log_step == 0:
                self.wrt_step = (epoch - 1) * len(self.train_loader) + batch_idx
                self.writer.add_scalar(f'{self.wrt_mode}/loss', loss.item(), self.wrt_step)

            # FOR EVAL
            seg_metrics = eval_metrics(output_1, target, self.num_classes)
            self._update_seg_metrics(*seg_metrics)
            pixAcc, mIoU, f1_mi, f1_ma, _ = self._get_seg_metrics().values()

            # PRINT INFO
            tbar.set_description(
                'TRAIN ({}) | Loss: {:.3f} | Acc {:.2f} mIoU {:.2f} f1_mi {:.2f} f1_ma {:.2f} | B {:.2f} D {:.2f} |'.format(
                    epoch, self.total_loss.average,
                    pixAcc, mIoU, f1_mi / (batch_idx + 1),
                                  f1_ma / (batch_idx + 1),
                    self.batch_time.average, self.data_time.average))

        # METRICS TO TENSORBOARD
        seg_metrics = self._get_seg_metrics()
        for k, v in list(seg_metrics.items())[:-1]:
            self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)
        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(f'{self.wrt_mode}/Learning_rate_{i}', opt_group['lr'], self.wrt_step)
            # self.writer.add_scalar(f'{self.wrt_mode}/Momentum_{k}', opt_group['momentum'], self.wrt_step)

        # RETURN LOSS & METRICS
        log = {'loss': self.total_loss.average,
               **seg_metrics}

        # if self.lr_scheduler is not None: self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        if self.val_loader is None:
            self.logger.warning('Not data loader was passed for the validation step, No validation is performed !')
            return {}
        self.logger.info('\n###### EVALUATION ######')

        self.model.eval()
        self.wrt_mode = 'val'

        self._reset_metrics()
        tbar = tqdm(self.val_loader, ncols=130)
        with torch.no_grad():
            val_visual = []
            for batch_idx, (data, target, edge_gt) in enumerate(tbar):
                data, target, edge_gt = data.to(self.device), target.to(self.device), edge_gt.to(self.device)
                # LOSS
                # output_1, edge_1, edge_2, edge_3, edge_4 = self.model(data)
                output_1, output_2, output_3, output_4, output_5, edge_1, edge_2, edge_3, edge_4 = self.model(data)
                # output_1, output_2, output_3, output_4, output_5, edge_1 = self.model(data)
                # output_1, output_2, output_3, output_4, output_5 = self.model(data)
                # output_1 = self.model(data)
                # loss = self.loss(output_1, target)
                loss1 = self.loss(output_1, target)
                loss2 = self.loss(output_2, target)
                loss3 = self.loss(output_3, target)
                loss4 = self.loss(output_4, target)
                loss5 = self.loss(output_5, target)
                loss = loss1 + loss2 + loss3 + loss4 + loss5
                # loss = loss1

                edge_3 = self.upsample_2(edge_3)
                edge_4 = self.upsample_3(edge_4)
                edge_loss_1 = CE(edge_1, edge_gt.unsqueeze(dim=1).float())
                edge_loss_2 = CE(edge_2, edge_gt.unsqueeze(dim=1).float())
                edge_loss_3 = CE(edge_3, edge_gt.unsqueeze(dim=1).float())
                edge_loss_4 = CE(edge_4, edge_gt.unsqueeze(dim=1).float())
                total_loss = loss + edge_loss_1 + edge_loss_2 + edge_loss_3 + edge_loss_4
                # total_loss = loss + edge_loss_1
                # total_loss = loss
                if isinstance(self.loss, torch.nn.DataParallel):
                    total_loss = loss.mean()
                self.total_loss.update(total_loss.item())

                seg_metrics = eval_metrics(output_1, target, self.num_classes)
                self._update_seg_metrics(*seg_metrics)

                # LIST OF IMAGE TO VIZ (15 images)
                if len(val_visual) < 15:
                    target_np = target.data.cpu().numpy()
                    output_np = output_1.data.max(1)[1].cpu().numpy()
                    val_visual.append([data[0].data.cpu(), target_np[0], output_np[0]])

                # PRINT INFO
                pixAcc, mIoU, f1_mi, f1_ma, _ = self._get_seg_metrics().values()
                tbar.set_description('EVAL ({}) | Loss: {:.3f}, PixelAcc: {:.2f}, Mean IoU: {:.2f}, f1_mi: {:.2f}, '
                                     'f1_ma: {:.2f} |'.format(epoch,
                                                              self.total_loss.average,
                                                              pixAcc, mIoU,
                                                              f1_mi / (batch_idx + 1), f1_ma / (batch_idx + 1)))

            # WRTING & VISUALIZING THE MASKS
            val_img = []
            palette = self.train_loader.dataset.palette
            for d, t, o in val_visual:
                d = self.restore_transform(d)
                t, o = colorize_mask(t, palette), colorize_mask(o, palette)
                d, t, o = d.convert('RGB'), t.convert('RGB'), o.convert('RGB')
                [d, t, o] = [self.viz_transform(x) for x in [d, t, o]]
                val_img.extend([d, t, o])
            val_img = torch.stack(val_img, 0)
            val_img = make_grid(val_img.cpu(), nrow=3, padding=5)
            self.writer.add_image(f'{self.wrt_mode}/inputs_targets_predictions', val_img, self.wrt_step)

            # METRICS TO TENSORBOARD
            self.wrt_step = (epoch) * len(self.val_loader)
            self.writer.add_scalar(f'{self.wrt_mode}/loss', self.total_loss.average, self.wrt_step)
            seg_metrics = self._get_seg_metrics()
            for k, v in list(seg_metrics.items())[:-1]:
                self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)

            log = {
                'val_loss': self.total_loss.average,
                **seg_metrics
            }

        return log

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.f1_mi = 0
        self.f1_ma = 0
        self.total_inter, self.total_union = 0, 0
        self.total_correct, self.total_label = 0, 0

    def _update_seg_metrics(self, correct, labeled, f1_mi, f1_ma, inter, union):
        self.total_correct += correct
        self.total_label += labeled
        self.f1_mi += f1_mi
        self.f1_ma += f1_ma
        self.total_inter += inter
        self.total_union += union

    def _get_seg_metrics(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        f1_mi = self.f1_mi
        f1_ma = self.f1_ma
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "F1_mi": np.round(f1_mi, 3),
            "F1_ma": np.round(f1_ma, 3),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))
        }
