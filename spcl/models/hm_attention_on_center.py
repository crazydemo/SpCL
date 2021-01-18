import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import init
from torch import nn, autograd


class HM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        # for x, y in zip(inputs[:inputs.size()[0]//2, :], indexes[:inputs.size()[0]//2]):
        for x, y in zip(inputs, indexes):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None

class HM_no_update(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        # for x, y in zip(inputs[:inputs.size()[0]//2, :], indexes[:inputs.size()[0]//2]):
        for x, y in zip(inputs, indexes):
            ctx.features[y] = ctx.features[y]

        return grad_inputs, None, None, None

def hm(inputs, indexes, features, momentum=0.5):
    return HM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))

def hm_no_update(inputs, indexes, features, momentum=0.5):
    return HM_no_update.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class HybridMemory(nn.Module):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2):
        super(HybridMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp

        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer('labels', torch.zeros(num_samples).long())

    def forward(self, inputs, indexes, train=True):
        # inputs: B*2048, features: L*2048
        inputs_raw = inputs
        if train:
            inputs = hm(inputs, indexes, self.features, self.momentum)
        else:
            inputs = hm_no_update(inputs, indexes, self.features, self.momentum)

        B = inputs.size(0)

        def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
            exps = torch.exp(vec)
            masked_exps = exps * mask.float().clone()
            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
            return (masked_exps/masked_sums)

        targets = self.labels[indexes].clone()
        labels = self.labels.clone()

        attn_mask = (targets.unsqueeze(1).repeat(1, labels.size()[0])==labels.unsqueeze(0).repeat(B, 1)).float()
        attn = attn_mask * inputs
        centers = attn.mm(self.features)
        attn_with_centers = inputs_raw.mm(centers.t()) / self.temp

        inputs /= self.temp
        sim = torch.zeros(labels.max()+1, B).float().cuda()
        sim.index_add_(0, labels, inputs.t().contiguous())
        nums = torch.zeros(labels.max()+1, 1).float().cuda()
        nums.index_add_(0, labels, torch.ones(self.num_samples,1).float().cuda())
        mask = (nums>0).float()
        sim /= (mask*nums+(1-mask)).clone().expand_as(sim)

        attn_mask = torch.zeros(labels.max()+1, B).float().cuda()
        attn_score = torch.zeros(labels.max()+1, B).float().cuda()
        for i in range(B):
            attn_mask[targets[i], i] = 1
            attn_score[targets[i], i] = attn_with_centers[i, i]
            # sim[targets[i], i] = attn_with_centers[i, i]

        sim = sim*(1-attn_mask)+attn_score

        mask = mask.expand_as(sim)
        masked_sim = masked_softmax(sim.t().contiguous(), mask.t().contiguous())
        return F.nll_loss(torch.log(masked_sim+1e-6), targets)
