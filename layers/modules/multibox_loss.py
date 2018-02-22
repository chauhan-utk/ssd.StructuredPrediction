import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import v2 as cfg
from ..box_utils import match, log_sum_exp, loss_aug_infer_loss, rematch, decode
from .boxloss import BoxLoss


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']
        self.epsilon = 0.1
        self.positive = True
        self.boxloss = BoxLoss()

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        # remove encoding from predicted boxes
        loc_data_decoded = torch.Tensor(num, num_priors, 4)
        y_direct = torch.Tensor(num, num_priors, 4)
        y_direct_conf = torch.LongTensor(num, num_priors)

        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            overlaps_intr = None # intermediate overlap values
            overlaps_intr = match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)
            loc_data_decoded[idx] = decode(loc_data[idx].data, defaults, self.variance)
            aug_loss = loss_aug_infer_loss(truths, labels, loc_data_decoded[idx], conf_data[idx])
            assert(overlaps_intr.size() == aug_loss.size()),"Expected same size got %s and %s" % (str(overlaps_intr.size()), str(aug_loss.size()))
            if self.positive:
                overlaps_intr = overlaps_intr + self.epsilon * aug_loss
            else:
                overlaps_intr = overlaps_intr - self.epsilon * aug_loss
            rematch(self.threshold, truths, defaults, labels, y_direct, y_direct_conf, overlaps_intr, idx)

        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            y_direct_conf = y_direct_conf.cuda()
            loc_data_decoded = loc_data_decoded.cuda()

        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        y_direct_conf = Variable(y_direct_conf, requires_grad=False)
        loc_data_decoded = Variable(loc_data_decoded, requires_grad=False)

        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)

        loc_t_classes = conf_t[pos] # [bbox without background label]

        # get predicted class for each prior box
        # class are 0 - 20
        pos_idx_class = pos.unsqueeze(pos.dim()).expand_as(conf_data)
        loc_p_classes = conf_data[pos_idx_class].data.view(-1, self.num_classes)
        # loc_p_classes, _ = loc_p_classes.max(1)
        _, loc_p_classes = loc_p_classes.max(1) # torch.LongTensor
        # loc_p_classes = 1 + loc_p_classes # check box_utils.py:L120

        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data_decoded)
        loc_p = loc_data_decoded[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)

        # loss augmented inference calculation
        pos_y = y_direct_conf > 0
        pos_y_idx = pos_y.unsqueeze(pos_y.dim()).expand_as(loc_data_decoded)
        loc_p_infr = loc_data_decoded[pos_y_idx].view(-1,4)

        if self.use_gpu:
            loc_p_classes = loc_p_classes.cuda()
            loc_t_classes = loc_t_classes.cuda()
        loc_p_classes = Variable(loc_p_classes, requires_grad=False)
        # loc_t_classes = Variable(loc_t_classes, requires_grad=False)
        self.boxloss.setdata(self.epsilon, self.positive, loc_p_infr) # set for backward

        loss_l = self.boxloss.apply(loc_p, loc_p_classes, loc_t, loc_t_classes)
        # loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)
        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)

        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c
