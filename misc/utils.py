from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

def apply_along_batch(func, M):
    #apply torch function for each image in a batch, and concatenate results back into a single tensor
    tensorList = [func(m) for m in torch.unbind(M, dim=0) ]
    result = torch.stack(tensorList, dim=0)
    return result

def if_use_att(caption_model):
    # Decide if load attention feature according to caption model
    if caption_model in ['show_tell', 'all_img', 'fc']:
        return False
    return True

# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix.item())]
            else:
                break
        out.append(txt)
    return out

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq>0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]

        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size=0, padding_idx=0, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False, reduce=False)
        # self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        # self.size = size
        self.true_dist = None

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]

        input = to_contiguous(input).view(-1, input.size(-1))
        target = to_contiguous(target).view(-1)
        mask = to_contiguous(mask).view(-1)

        # assert x.size(1) == self.size
        self.size = input.size(1)
        # true_dist = x.data.clone()
        true_dist = input.data.clone()
        # true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.fill_(self.smoothing / (self.size - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # true_dist[:, self.padding_idx] = 0
        # mask = torch.nonzero(target.data == self.padding_idx)
        # self.true_dist = true_dist
        return (self.criterion(input, true_dist).sum(1) * mask).sum() / mask.sum()

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def get_lr(optimizer):
    for group in optimizer.param_groups:
        return group['lr']

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

def build_optimizer(params, opt):
    if opt.optim == 'rmsprop':
        return optim.RMSprop(params, opt.learning_rate, opt.optim_alpha, opt.optim_epsilon, weight_decay=opt.weight_decay)
    elif opt.optim == 'adagrad':
        return optim.Adagrad(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgd':
        return optim.SGD(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdm':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdmom':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay, nesterov=True)
    elif opt.optim == 'adam':
        return optim.Adam(params, opt.learning_rate, (opt.optim_alpha, opt.optim_beta), opt.optim_epsilon, weight_decay=opt.weight_decay)
    else:
        raise Exception("bad option opt.optim: {}".format(opt.optim))


class NoamOpt(object):
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def __getattr__(self, name):
        return getattr(self.optimizer, name)

class ReduceLROnPlateau(object):
    "Optim wrapper that implements rate."
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08):
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode, factor, patience, verbose, threshold, threshold_mode, cooldown, min_lr, eps)
        self.optimizer = optimizer
        self.current_lr = get_lr(optimizer)

    def step(self):
        "Update parameters and rate"
        self.optimizer.step()

    def scheduler_step(self, val):
        self.scheduler.step(val)
        self.current_lr = get_lr(self.optimizer)

    def state_dict(self):
        return {'current_lr':self.current_lr,
                'scheduler_state_dict': {key: value for key, value in self.scheduler.__dict__.items() if key not in {'optimizer', 'is_better'}},
                'optimizer_state_dict': self.optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        if 'current_lr' not in state_dict:
            # it's normal optimizer
            self.optimizer.load_state_dict(state_dict)
            set_lr(self.optimizer, self.current_lr) # use the lr fromt the option
        else:
            # it's a schduler
            self.current_lr = state_dict['current_lr']
            self.scheduler.__dict__.update(state_dict['scheduler_state_dict'])
            self.scheduler._init_is_better(mode=self.scheduler.mode, threshold=self.scheduler.threshold, threshold_mode=self.scheduler.threshold_mode)
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            # current_lr is actually useless in this case

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def __getattr__(self, name):
        return getattr(self.optimizer, name)

def get_std_opt(model, factor=1, warmup=2000):
    # return NoamOpt(model.tgt_embed[0].d_model, 2, 4000,
    #         torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    return NoamOpt(model.model.tgt_embed[0].d_model, factor, warmup,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

def BoxesRelationalEmbedding(f_g, dim_g=64, wave_len=1000):
    return (apply_along_batch(BoxRelationalEmbedding, f_g))
def BoxRelationalEmbedding(f_g, dim_g=64, wave_len=1000):
    #returns a relational embedding for each pair of bboxes, with dimension = dim_g
    #follow implementation of https://github.com/heefe92/Relation_Networks-pytorch/blob/master/model.py#L1014-L1055
    x_min, y_min, x_max, y_max = torch.chunk(f_g, 4, dim=1)

    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.

    #cx.view(1,-1) transposes the vector cx, and so dim(delta_x) = (dim(cx), dim(cx))
    delta_x = cx - cx.view(1, -1)
    delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
    delta_x = torch.log(delta_x)

    delta_y = cy - cy.view(1, -1)
    delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
    delta_y = torch.log(delta_y)

    delta_w = torch.log(w / w.view(1, -1))
    delta_h = torch.log(h / h.view(1, -1))
    size = delta_h.size()

    delta_x = delta_x.view(size[0], size[1], 1)
    delta_y = delta_y.view(size[0], size[1], 1)
    delta_w = delta_w.view(size[0], size[1], 1)
    delta_h = delta_h.view(size[0], size[1], 1)

    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)

    feat_range = torch.arange(dim_g / 8).cuda()
    dim_mat = feat_range / (dim_g / 8)
    dim_mat = 1. / (torch.pow(wave_len, dim_mat))

    dim_mat = dim_mat.view(1, 1, 1, -1)
    position_mat = position_mat.view(size[0], size[1], 4, -1)
    position_mat = 100. * position_mat

    mul_mat = position_mat * dim_mat
    mul_mat = mul_mat.view(size[0], size[1], -1)
    sin_mat = torch.sin(mul_mat)
    cos_mat = torch.cos(mul_mat)
    embedding = torch.cat((sin_mat, cos_mat), -1)

    return embedding

def get_box_feats(boxes, d):
    h,w = boxes.shape[:2]
    boxes_times_d = (d*boxes).astype(np.int32)
    boxes_wmin = boxes_times_d[:,:,0]
    boxes_wmax = boxes_times_d[:,:,2]
    boxes_hmin = boxes_times_d[:,:,1]
    boxes_hmax = boxes_times_d[:,:,3]

    box_hfeats = np.zeros((h,w,d))
    for i in range(h):
        for j in range(w):
            if not np.all(boxes_times_d[i,j]==np.zeros(4)):
                h_vector = np.concatenate([np.zeros(boxes_hmin[i,j]), np.ones(boxes_hmax[i,j]-boxes_hmin[i,j]), np.zeros(d-boxes_hmax[i,j])])
                box_hfeats[i,j]+=h_vector

    box_wfeats = np.zeros((h,w,d))
    for i in range(h):
        for j in range(w):
            if not np.all(boxes_times_d[i,j]==np.zeros(4)):
                w_vector = np.concatenate([np.zeros(boxes_wmin[i,j]), np.ones(boxes_wmax[i,j]-boxes_wmin[i,j]), np.zeros(d-boxes_wmax[i,j])])
                box_wfeats[i,j]+=w_vector
    return(box_hfeats, box_wfeats)

def single_image_get_box_feats(boxes, d):
    h = boxes.shape[0]
    boxes_times_d = (d*boxes).astype(np.int32)
    boxes_wmin = boxes_times_d[:,0]
    boxes_wmax = boxes_times_d[:,2]
    boxes_hmin = boxes_times_d[:,1]
    boxes_hmax = boxes_times_d[:,3]

    box_hfeats = np.zeros((h,d))
    for i in range(h):
        #for j in range(w):
            if not np.all(boxes_times_d[i]==np.zeros(4)):
                h_vector = np.concatenate([np.zeros(boxes_hmin[i]), np.ones(boxes_hmax[i]-boxes_hmin[i]), np.zeros(d-boxes_hmax[i])])
                box_hfeats[i]+=h_vector

    box_wfeats = np.zeros((h,d))
    for i in range(h):
        #for j in range(w):
            if not np.all(boxes_times_d[i]==np.zeros(4)):
                w_vector = np.concatenate([np.zeros(boxes_wmin[i]), np.ones(boxes_wmax[i]-boxes_wmin[i]), np.zeros(d-boxes_wmax[i])])
                box_wfeats[i]+=w_vector
    return(box_hfeats, box_wfeats)

def get_box_areas(arr):
    return((arr[:,2]-arr[:,0])*(arr[:,3]-arr[:,1]))

def torch_get_box_feats(boxes, d):
    device = boxes.device
    h,w = boxes.shape[:2]
    boxes_times_d = (d*boxes).type(torch.int32)
    boxes_wmin = boxes_times_d[:,:,0]
    boxes_wmax = boxes_times_d[:,:,2]
    boxes_hmin = boxes_times_d[:,:,1]
    boxes_hmax = boxes_times_d[:,:,3]

    box_hfeats = torch.zeros((h,w,d), device=device)
    zero_fourtuple=torch.zeros(4,dtype=torch.int32,device=device)

    for i in range(h):
        for j in range(w):
            if not torch.all(boxes_times_d[i,j]==zero_fourtuple):
                h_vector = torch.cat([torch.zeros(boxes_hmin[i,j], device=device), torch.ones(boxes_hmax[i,j]-boxes_hmin[i,j], device=device), torch.zeros(d-boxes_hmax[i,j], device=device)])
                box_hfeats[i,j]+=h_vector

    box_wfeats = torch.zeros((h,w,d), device=device)
    for i in range(h):
        for j in range(w):
            if not all(boxes_times_d[i,j]==zero_fourtuple):
                w_vector = torch.cat([torch.zeros(boxes_wmin[i,j], device=device), torch.ones(boxes_wmax[i,j]-boxes_wmin[i,j], device=device), torch.zeros(d-boxes_wmax[i,j], device=device)])
                box_wfeats[i,j]+=w_vector
    return(box_hfeats, box_wfeats)
