import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

def to_contiguous(tensor):
  if tensor.is_contiguous():
    return tensor
  else:
    return tensor.contiguous()

def _assert_no_grad(variable):
  assert not variable.requires_grad, \
    "nn criterions don't compute the gradient w.r.t. targets - please " \
    "mark these variables as not requiring gradients"

class SequenceCrossEntropyLoss(nn.Module):
  def __init__(self, 
               weight=None,
               size_average=True,
               ignore_index=-100,
               sequence_normalize=False,
               sample_normalize=True):
    super(SequenceCrossEntropyLoss, self).__init__()
    self.weight = weight
    self.size_average = size_average
    self.ignore_index = ignore_index
    self.sequence_normalize = sequence_normalize
    self.sample_normalize = sample_normalize

    assert (sequence_normalize and sample_normalize) == False

  def forward(self, input, target, length):
    _assert_no_grad(target)
    # print("input: ", input.size())
    # print("target: ", target[:, :max(length)])
    # print("size input", input.size())
    # length to mask
    batch_size, def_max_length = target.size(0), target.size(1)
    # print("length size: ",length.size())
    mask = torch.zeros(batch_size, def_max_length)
    for i in range(batch_size):
      mask[i,:length[i]].fill_(1)
    mask = mask.type_as(input)
    # truncate to the same size
    max_length = max(length)
    # print("max length: ", max_length)
    assert max_length == input.size(1)
    target = target[:, :max_length]
    mask =  mask[:, :max_length]
    # print(mask)
    input = to_contiguous(input).view(-1, input.size(2))
    input = F.log_softmax(input, dim=1)
    target = to_contiguous(target).view(-1, 1)
    mask = to_contiguous(mask).view(-1, 1)
    output = - input.gather(1, target.long()) * mask
    #focal ctc loss
    anpha = 0.99
    gama = 1

    output = torch.sum(output)
    if self.sequence_normalize:
      output = output / torch.sum(mask)
    if self.sample_normalize:
      output = output / batch_size

    return output