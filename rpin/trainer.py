import os
import torch
import numpy as np
import torch.nn.functional as F
from rpin.utils.misc import tprint
from timeit import default_timer as timer
from rpin.utils.config import _C as C
from rpin.utils.bbox import xyxy_to_rois, xywh2xyxy
from rpin.utils.vis import plot_rollouts
import shutil


class Trainer(object):
    def __init__(self, device, train_loader, val_loader, model, optim,
                 max_iters, num_gpus, logger, output_dir):
        # misc
        self.device = device
        self.output_dir = output_dir
        self.logger = logger
        self.num_gpus = num_gpus
        # data loading
        self.train_loader, self.val_loader = train_loader, val_loader
        # nn optimization
        self.model = model
        self.optim = optim
        # input setting
        self.input_size = C.RPIN.INPUT_SIZE
        self.ptrain_size, self.ptest_size = C.RPIN.PRED_SIZE_TRAIN, C.RPIN.PRED_SIZE_TEST
        self.input_height, self.input_width = C.RPIN.INPUT_HEIGHT, C.RPIN.INPUT_WIDTH
        self.batch_size = C.SOLVER.BATCH_SIZE
        # train loop settings
        self.iterations = 0
        self.epochs = 0
        self.max_iters = max_iters
        self.val_interval = C.SOLVER.VAL_INTERVAL
        self.fg_correct, self.bg_correct, self.fg_num, self.bg_num = 0, 0, 0, 0
        # loss settings
        self._setup_loss()
        # timer setting
        self.best_mean = 1e6
        # plot image
        self.plot_image = 10000

    def train(self):
        print_msg = "| ".join(["progress  | mean "] + list(map("{:6}".format, self.loss_name)))
        self.model.train()
        print('\r', end='')
        self.logger.info(print_msg)
        while self.iterations < self.max_iters:
            self.train_epoch()
            self.epochs += 1

    def train_epoch(self):
        for batch_idx, (data, data_t, rois, gt_boxes, gt_masks, valid, g_idx, seq_l) in enumerate(self.train_loader):
            self._adjust_learning_rate()
            data, data_t = data.to(self.device), data_t.to(self.device)
            rois = xyxy_to_rois(rois, batch=data.shape[0], time_step=data.shape[1], num_devices=self.num_gpus)
            self.optim.zero_grad()

            outputs = self.model(data, rois, num_rollouts=self.ptrain_size, g_idx=g_idx, x_t=data_t, phase='train')
            labels = {
                'boxes': gt_boxes.to(self.device),
                'masks': gt_masks.to(self.device),
                'valid': valid.to(self.device),
                'seq_l': seq_l.to(self.device),
            }
            loss = self.loss(outputs, labels, 'train')
            loss.backward()
            self.optim.step()
            # this is an approximation for printing; the dataset size may not divide the batch size
            self.iterations += self.batch_size

            print_msg = ""
            print_msg += f"{self.epochs:03}/{self.iterations // 1000:04}k"
            print_msg += f" | "
            mean_loss = np.mean(np.array(self.box_p_step_losses[:self.ptrain_size]) / self.loss_cnt) * 1e3
            print_msg += f"{mean_loss:.3f} | "
            print_msg += f" | ".join(
                ["{:.3f}".format(self.losses[name] * 1e3 / self.loss_cnt) for name in self.loss_name])
            if C.RPIN.SEQ_CLS_LOSS_WEIGHT:
                print_msg += f" | {self.fg_correct / self.fg_num:.3f} | {self.bg_correct / self.bg_num:.3f}"
            speed = self.loss_cnt / (timer() - self.time)
            eta = (self.max_iters - self.iterations) / speed / 3600
            print_msg += f" | speed: {speed:.1f} | eta: {eta:.2f} h"
            # terminal_width = os.get_terminal_size().columns
            terminal_width = shutil.get_terminal_size().columns
            print_msg += (" " * (terminal_width - len(print_msg) - 10))
            tprint(print_msg)

            if self.iterations % self.val_interval == 0:
                print("Evaluating...")
                self.snapshot()
                self.val()
                self._init_loss()
                self.model.train()

            if self.iterations >= self.max_iters:
                print('\r', end='')
                print(f'{self.best_mean:.3f}')
                break

    def val(self):
        self.model.eval()
        self._init_loss()

        if C.RPIN.VAE:
            losses = dict.fromkeys(self.loss_name, 0.0)
            box_p_step_losses = [0.0 for _ in range(self.ptest_size)]
            masks_step_losses = [0.0 for _ in range(self.ptest_size)]

        for batch_idx, (data, _, rois, gt_boxes, gt_masks, valid, g_idx, seq_l) in enumerate(self.val_loader):
            tprint(f'eval: {batch_idx}/{len(self.val_loader)}')
            with torch.no_grad():

                data = data.to(self.device)
                rois = xyxy_to_rois(rois, batch=data.shape[0], time_step=data.shape[1], num_devices=self.num_gpus)
                labels = {
                    'boxes': gt_boxes.to(self.device),
                    'masks': gt_masks.to(self.device),
                    'valid': valid.to(self.device),
                    'seq_l': seq_l.to(self.device),
                }

                outputs = self.model(data, rois, num_rollouts=self.ptest_size, g_idx=g_idx, phase='test')
                self.loss(outputs, labels, 'test')
                # VAE multiple runs
                if C.RPIN.VAE:
                    vae_best_mean = np.mean(np.array(self.box_p_step_losses[:self.ptest_size]) / self.loss_cnt) * 1e3
                    losses_t = self.losses.copy()
                    box_p_step_losses_t = self.box_p_step_losses.copy()
                    masks_step_losses_t = self.masks_step_losses.copy()
                    for i in range(9):
                        outputs = self.model(data, rois, num_rollouts=self.ptest_size, g_idx=g_idx, phase='test')
                        self.loss(outputs, labels, 'test')
                        mean_loss = np.mean(np.array(self.box_p_step_losses[:self.ptest_size]) / self.loss_cnt) * 1e3
                        if mean_loss < vae_best_mean:
                            losses_t = self.losses.copy()
                            box_p_step_losses_t = self.box_p_step_losses.copy()
                            masks_step_losses_t = self.masks_step_losses.copy()
                            vae_best_mean = mean_loss
                        self._init_loss()

                    for k, v in losses.items():
                        losses[k] += losses_t[k]
                    for i in range(len(box_p_step_losses)):
                        box_p_step_losses[i] += box_p_step_losses_t[i]
                        masks_step_losses[i] += masks_step_losses_t[i]

            if self.plot_image > 0:
                outputs = {
                    'boxes': outputs['boxes'].cpu().numpy(),
                    'masks': outputs['masks'].cpu().numpy() if C.RPIN.MASK_LOSS_WEIGHT else None,
                }
                outputs['boxes'][..., 0::2] *= self.input_width
                outputs['boxes'][..., 1::2] *= self.input_height
                outputs['boxes'] = xywh2xyxy(
                    outputs['boxes'].reshape(-1, 4)
                ).reshape((data.shape[0], -1, C.RPIN.MAX_NUM_OBJS, 4))

                labels = {
                    'boxes': labels['boxes'].cpu().numpy(),
                    'masks': labels['masks'].cpu().numpy(),
                }
                labels['boxes'][..., 0::2] *= self.input_width
                labels['boxes'][..., 1::2] *= self.input_height
                labels['boxes'] = xywh2xyxy(
                    labels['boxes'].reshape(-1, 4)
                ).reshape((data.shape[0], -1, C.RPIN.MAX_NUM_OBJS, 4))

                for i in range(rois.shape[0]):
                    batch_size = C.SOLVER.BATCH_SIZE if not C.RPIN.VAE else 1
                    plot_image_idx = batch_size * batch_idx + i
                    if plot_image_idx%self.plot_image == 0:
                        tprint(f'plotting: {plot_image_idx}' + ' ' * 20)
                        video_idx, img_idx = self.val_loader.dataset.video_info[plot_image_idx]
                        video_name = self.val_loader.dataset.video_list[video_idx]

                        v = valid[i].numpy().astype(bool)
                        pred_boxes_i = outputs['boxes'][i][:, v]
                        gt_boxes_i = labels['boxes'][i][:, v]

                        im_data = np.load(video_name)
                        bg_image = None
                        video_name_aux = video_name.split('/')
                        output_name = video_name_aux[-2] + '_' + video_name_aux[-1].replace('.npy', '')

                        scale_w = im_data.shape[1] / self.input_width
                        scale_h = im_data.shape[0] / self.input_height
                        pred_boxes_i[..., [0, 2]] *= scale_w
                        pred_boxes_i[..., [1, 3]] *= scale_h
                        gt_boxes_i[..., [0, 2]] *= scale_w
                        gt_boxes_i[..., [1, 3]] *= scale_h

                        pred_masks_i = None
                        if C.RPIN.MASK_LOSS_WEIGHT:
                            pred_masks_i = outputs['masks'][i][:, v]

                        ouput_dir = os.path.join(self.output_dir, 'figures', str(self.iterations))

                        plot_rollouts(im_data, pred_boxes_i, gt_boxes_i,
                                      pred_masks_i, labels['masks'][i][:, v],
                                      output_dir=ouput_dir, output_name=output_name, bg_image=bg_image)

        if C.RPIN.VAE:
            self.losses = losses.copy()
            self.box_p_step_losses = box_p_step_losses.copy()
            self.loss_cnt = len(self.val_loader)

        print('\r', end='')
        print_msg = ""
        print_msg += f"{self.epochs:03}/{self.iterations // 1000:04}k"
        print_msg += f" | "
        mean_loss = np.mean(np.array(self.box_p_step_losses[:self.ptest_size]) / self.loss_cnt) * 1e3
        print_msg += f"{mean_loss:.3f} | "

        if mean_loss < self.best_mean:
            self.snapshot('ckpt_best.path.tar')
            self.best_mean = mean_loss

        print_msg += f" | ".join(["{:.3f}".format(self.losses[name] * 1e3 / self.loss_cnt) for name in self.loss_name])
        if C.RPIN.SEQ_CLS_LOSS_WEIGHT:
            print_msg += f" | {self.fg_correct / (self.fg_num + 1e-9):.3f} | {self.bg_correct / (self.bg_num + 1e-9):.3f}"
        # terminal_width = os.get_terminal_size().columns
        terminal_width = shutil.get_terminal_size().columns
        print_msg += (" " * (terminal_width - len(print_msg) - 10))
        self.logger.info(print_msg)

    def loss(self, outputs, labels, phase):
        self.loss_cnt += labels['boxes'].shape[0]
        pred_size = eval(f'self.p{phase}_size')
        # calculate bbox loss
        # of shape (batch, time, #obj, 4)
        loss = (outputs['boxes'] - labels['boxes']) ** 2
        # take weighted sum over axis 2 (objs dim) since some index are not valid
        valid = labels['valid'][:, None, :, None]
        loss = loss * valid
        loss = loss.sum(2) / valid.sum(2)
        loss *= self.position_loss_weight

        for i in range(pred_size):
            self.box_p_step_losses[i] += loss[:, i, :2].sum().item()
            self.box_s_step_losses[i] += loss[:, i, 2:].sum().item()

        self.losses['p_1'] = float(np.mean(self.box_p_step_losses[:self.ptrain_size]))
        self.losses['p_2'] = float(np.mean(self.box_p_step_losses[self.ptrain_size:])) \
            if self.ptrain_size < self.ptest_size else 0
        self.losses['s_1'] = float(np.mean(self.box_s_step_losses[:self.ptrain_size]))
        self.losses['s_2'] = float(np.mean(self.box_s_step_losses[self.ptrain_size:])) \
            if self.ptrain_size < self.ptest_size else 0

        mask_loss = 0
        if C.RPIN.MASK_LOSS_WEIGHT > 0:
            # of shape (batch, time, #obj, m_sz, m_sz)
            mask_loss_ = F.binary_cross_entropy(outputs['masks'], labels['masks'], reduction='none')
            mask_loss = mask_loss_.mean((3, 4))
            valid = labels['valid'][:, None, :]
            mask_loss = mask_loss * valid
            mask_loss = mask_loss.sum(2) / valid.sum(2)

            for i in range(pred_size):
                self.masks_step_losses[i] += mask_loss[:, i].sum().item()

            m1_loss = self.masks_step_losses[:self.ptrain_size]
            m2_loss = self.masks_step_losses[self.ptrain_size:]
            self.losses['m_1'] = np.mean(m1_loss)
            self.losses['m_2'] = np.mean(m2_loss) if self.ptrain_size < self.ptest_size else 0

            mask_loss = mask_loss.mean(0)
            init_tau = C.RPIN.DISCOUNT_TAU ** (1 / self.ptrain_size)
            tau = init_tau + (self.iterations / self.max_iters) * (1 - init_tau)
            tau = torch.pow(tau, torch.arange(pred_size, out=torch.FloatTensor()))[:, None].to('cuda')
            mask_loss = ((mask_loss * tau) / tau.sum(axis=0, keepdims=True)).sum()
            mask_loss = mask_loss * C.RPIN.MASK_LOSS_WEIGHT

        seq_loss = 0
        if C.RPIN.SEQ_CLS_LOSS_WEIGHT > 0:
            seq_loss = F.binary_cross_entropy(outputs['score'], labels['seq_l'], reduction='none')
            self.losses['seq'] += seq_loss.sum().item()
            seq_loss = seq_loss.mean() * C.RPIN.SEQ_CLS_LOSS_WEIGHT
            # calculate accuracy
            s = (outputs['score'] >= 0.5).eq(labels['seq_l'])
            fg_correct = s[labels['seq_l'] == 1].sum().item()
            bg_correct = s[labels['seq_l'] == 0].sum().item()
            fg_num = (labels['seq_l'] == 1).sum().item()
            bg_num = (labels['seq_l'] == 0).sum().item()
            self.fg_correct += fg_correct
            self.bg_correct += bg_correct
            self.fg_num += fg_num
            self.bg_num += bg_num

        kl_loss = 0
        if C.RPIN.VAE and phase == 'train':
            kl_loss = outputs['kl']
            self.losses['kl'] += kl_loss.sum().item()
            kl_loss = C.RPIN.VAE_KL_LOSS_WEIGHT * kl_loss.sum()

        # no need to do precise batch statistics, just do mean for backward gradient
        loss = loss.mean(0)
        init_tau = C.RPIN.DISCOUNT_TAU ** (1 / self.ptrain_size)
        tau = init_tau + (self.iterations / self.max_iters) * (1 - init_tau)
        tau = torch.pow(tau, torch.arange(pred_size, out=torch.FloatTensor()))[:, None].to('cuda')
        loss = ((loss * tau) / tau.sum(axis=0, keepdims=True)).sum()
        loss = loss + mask_loss + kl_loss + seq_loss

        return loss

    def snapshot(self, name='ckpt_latest.path.tar'):
        torch.save(
            {
                'arch': self.model.__class__.__name__,
                'model': self.model.state_dict(),
            },
            os.path.join(self.output_dir, name),
        )

    def _setup_loss(self):
        self.loss_name = []
        self.position_loss_weight = C.RPIN.POSITION_LOSS_WEIGHT
        self.loss_name += ['p_1', 'p_2', 's_1', 's_2']
        if C.RPIN.MASK_LOSS_WEIGHT:
            self.loss_name += ['m_1', 'm_2']
        if C.RPIN.VAE:
            self.loss_name += ['kl']
        if C.RPIN.SEQ_CLS_LOSS_WEIGHT:
            self.loss_name += ['seq']
        self._init_loss()

    def _init_loss(self):
        self.losses = dict.fromkeys(self.loss_name, 0.0)
        self.box_p_step_losses = [0.0 for _ in range(self.ptest_size)]
        self.box_s_step_losses = [0.0 for _ in range(self.ptest_size)]
        self.masks_step_losses = [0.0 for _ in range(self.ptest_size)]
        # an statistics of each validation
        self.fg_correct, self.bg_correct, self.fg_num, self.bg_num = 0, 0, 0, 0
        self.loss_cnt = 0
        self.time = timer()

    def _adjust_learning_rate(self):
        if self.iterations <= C.SOLVER.WARMUP_ITERS:
            lr = C.SOLVER.BASE_LR * self.iterations / C.SOLVER.WARMUP_ITERS
        else:
            if C.SOLVER.SCHEDULER == 'step':
                lr = C.SOLVER.BASE_LR
                for m_iters in C.SOLVER.LR_MILESTONES:
                    if self.iterations > m_iters:
                        lr *= C.SOLVER.LR_GAMMA
            elif C.SOLVER.SCHEDULER == 'cosine':
                lr = 0.5 * C.SOLVER.BASE_LR * (1 + np.cos(np.pi * self.iterations / self.max_iters))
            else:
                raise NotImplementedError

        for param_group in self.optim.param_groups:
            param_group['lr'] = lr
