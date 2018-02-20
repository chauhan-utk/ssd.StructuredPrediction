import torch
from torch.autograd import Function

class BoxLoss(Function):
    """Implement structured loss for predicting boxes
    """
    def __init__():
        self.epsilon = None
        self.positive = None

    def setdata(self, epsilon, positive, loc_p_infr):
        """
        loc_p_infr shape : torch.size([obj_not_bg, 4])
        """
        self.loc_data = loc_data
        self.conf_data = conf_data
        self.loc_p_infr = loc_p_infr

    def intersect(box_a, box_b):
        """
        copied from box_utils.py
        """
        A = box_a.size(0)
        B = box_a.size(0)
        max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                           box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                           box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)
        return inter[:, :, 0] * inter[:, :, 1]

    def jaccard(box_a, box_b):
        """
        copied from box_utils.py
        """
        inter = intersect(box_a, box_b)
        area_a = ((box_a[:, 2]-box_a[:, 0]) *
                  (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        area_b = ((box_b[:, 2]-box_b[:, 0]) *
                  (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
        union = area_a + area_b - inter
        return inter / union  # [A,B]

    def forward(self, loc_p, loc_p_classes, loc_t, loc_t_classes):
        """
        loc_p, loc_t : torch.FloatTensor
        loc_p_classes, loc_t_classes : torch.LongTensor
        class labels : [1 - 20]
        """
        assert (loc_p.size(0) == loc_t.size(0) == loc_p_classes.size(0) == loc_t_classes.size(0)),"Tensors dim do not match"
        assert(loc_p.size() == loc_t.size()),"Tensors size do not match"
        self.loc_p = loc_p
        eq = loc_p_classes == loc_t_classes
        neq = 1 - eq
        res = torch.Tensor(loc_t.size(0)).zero_()
        res.index_fill_(0,torch.nonzero(neq).squeeze_(1),1.0)
        for i in torch.nonzero(eq):
            res[i] = 1. - jaccard(loc_p[i,:].unsqueeze(0), loc_t[i,:].unsqueeze(0)).squeeze_()[0]
        return res.sum()

    def backward(self, grad_output):
        assert(self.loc_p_infr.size(0)>=lf.loc_p.size(0)),"loc_p_infr %s and loc_p %s not compatible" % (str(self.loc_p_infr.size()),str(self.loc_p.size()))

        grad_out = self.loc_p_infr[:self.loc_p.size(0),:] - self.loc_p
        grad_out /= self.epsilon
        if not self.positive:
            grad_out *= -1
        return grad_output * grad_out, None, None, None
