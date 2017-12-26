import torch
import numpy as np
from model.config import cfg
from model.bbox_transform import bbox_transform_inv, clip_boxes
from torch.autograd import Variable

def proposal_layer_zhbli(rpn_cls_prob, rpn_bbox_pred, im_info, _feat_stride, anchors, num_anchors):
    """zhbli
    Select the RoIs whose scores > 0.5
    """

    scores = rpn_cls_prob[:, :, :, num_anchors:]
    rpn_bbox_pred = rpn_bbox_pred.view(-1, 4)
    scores = scores.contiguous().view(-1, 1)

    top_inds = scores.sort(0, descending=True)[1] # Shape=[?,1]
    top_inds = top_inds.view(-1) # Change shape into [?]
    temp1 = scores[top_inds.data]<0.5
    temp2 = temp1.view(-1)
    temp3 = temp2.data.cpu().numpy()
    trunc_ind = np.where(temp3==1)[0][0]
    top_inds = top_inds[:trunc_ind] # shape= ?*1
    top_inds = top_inds.view(-1)

    # Do the selection here
    anchors = anchors[top_inds.data, :].contiguous()
    rpn_bbox_pred = rpn_bbox_pred[top_inds.data, :].contiguous()
    scores = scores[top_inds.data].contiguous()

    # Convert anchors into proposals via bbox transformations
    proposals = bbox_transform_inv(anchors, rpn_bbox_pred)

    # Clip predicted boxes to image
    proposals = clip_boxes(proposals, im_info[:2])

    # Output rois blob
    # Our RPN implementation only supports a single input image, so all
    # batch inds are 0
    batch_inds = Variable(proposals.data.new(proposals.size(0), 1).zero_())
    blob = torch.cat([batch_inds, proposals], 1)
    return blob, scores


