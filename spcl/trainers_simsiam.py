from __future__ import print_function, absolute_import
import time
import numpy as np
import collections

import torch
import torch.nn as nn
from torch.nn import functional as F

from .utils.meters import AverageMeter


class SpCLTrainer_UDA(object):
    def __init__(self, encoder, memory, source_classes):
        super(SpCLTrainer_UDA, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.source_classes = source_classes

    def train(self, epoch, data_loader_source, data_loader_target,
                    optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_s = AverageMeter()
        losses_t = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            source_inputs = data_loader_source.next()
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            s_inputs, s_targets, _ = self._parse_data(source_inputs)
            t_inputs, _, t_indexes = self._parse_data(target_inputs)

            # arrange batch for domain-specific BN
            device_num = torch.cuda.device_count()
            B, C, H, W = s_inputs.size()
            def reshape(inputs):
                return inputs.view(device_num, -1, C, H, W)
            s_inputs, t_inputs = reshape(s_inputs), reshape(t_inputs)
            inputs = torch.cat((s_inputs, t_inputs), 1).view(-1, C, H, W)

            # forward
            f_out = self._forward(inputs)

            # de-arrange batch
            f_out = f_out.view(device_num, -1, f_out.size(-1))
            f_out_s, f_out_t = f_out.split(f_out.size(1)//2, dim=1)
            f_out_s, f_out_t = f_out_s.contiguous().view(-1, f_out.size(-1)), f_out_t.contiguous().view(-1, f_out.size(-1))

            # compute loss with the hybrid memory
            loss_s = self.memory(f_out_s, s_targets)
            loss_t = self.memory(f_out_t, t_indexes+self.source_classes)

            loss = loss_s+loss_t
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_s.update(loss_s.item())
            losses_t.update(loss_t.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_s {:.3f} ({:.3f})\t'
                      'Loss_t {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_s.val, losses_s.avg,
                              losses_t.val, losses_t.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)


class SpCLTrainer_USL(object):
    def __init__(self, encoder, memory1, memory2):
        super(SpCLTrainer_USL, self).__init__()
        self.encoder = encoder
        self.memory1 = memory1
        self.memory2 = memory2

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs1 = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            x1, x2, _, indexes = self._parse_data(inputs1)

            inputs = torch.cat([x1, x2], 0)

            # forward
            '''image1'''
            f_out = self._forward(inputs)
            '''image1 with label1, 2; image2 with label1, 2'''
            z11, p12, z21, p22 = f_out[:64, :2048], f_out[:64, 2048:], f_out[64:, :2048], f_out[64:, 2048:]

            '''for image1'''
            loss1 = self.memory1(z11, indexes)
            loss2 = self.memory2(z11, indexes, train=False)
            loss3 = self.memory2(p12, indexes)
            loss4 = self.memory1(p12, indexes, train=False)
            loss_ml1 = loss1 + 0.1 * loss2 + loss3 + 0.1 * loss4

            '''for image2'''
            loss1 = self.memory1(z21, indexes)
            loss2 = self.memory2(z21, indexes, train=False)
            loss3 = self.memory2(p22, indexes)
            loss4 = self.memory1(p22, indexes, train=False)
            loss_ml2 = loss1 + 0.1 * loss2 + loss3 + 0.1 * loss4

            '''mutual learning'''
            loss_siam = -0.5*(p22*z11.detach()).sum(1).mean()-0.5*(p12*z21.detach()).sum(1).mean()

            loss = 0.5 * loss_ml1 + 0.5*loss_ml2 + 1.0 *loss_siam

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        x1, x2 = imgs
        return x1.cuda(), x2.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)
