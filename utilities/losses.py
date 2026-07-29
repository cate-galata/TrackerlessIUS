import torch
from torch import nn
import torch.nn.functional as F
      
class DC(nn.Module):
    def __init__(self,nb_classes):
        super(DC, self).__init__()
        
        self.softmax = nn.Softmax(1)
        self.nb_classes = nb_classes

    @staticmethod 
    def onehot(gt,shape):
        shp_y = gt.shape
        gt = gt.long()
        y_onehot = torch.zeros(shape)
        y_onehot = y_onehot.cuda()
        y_onehot.scatter_(1, gt, 1)
        return y_onehot


    def dice(self, output, target):
        output = self.softmax(output)
        if not all([i == j for i, j in zip(output.shape, target.shape)]):
            target = self.onehot(target, output.shape)

        sum_axis = list(range(2,len(target.shape)))

        s = (10e-20)
        intersect = torch.sum(output * target,sum_axis)
        dice = (2 * intersect) / (torch.sum(output,sum_axis) + torch.sum(target,sum_axis) + s)
        #dice shape is (batch_size, nb_classes)
        return 1.0 - dice.mean()  

    def forward(self, output, target):
        result = self.dice(output, target)
        return result

class DC_SOFT(nn.Module):
    def __init__(self,nb_classes):
        super(DC_SOFT, self).__init__()
        
        self.softmax = nn.Softmax(1)
        self.nb_classes = nb_classes

    @staticmethod 
    def onehot(gt,shape):
        shp_y = gt.shape
        gt = gt.long()
        y_onehot = torch.zeros(shape)
        y_onehot = y_onehot.cuda()
        y_onehot.scatter_(1, gt, 1)
        return y_onehot


    # def dice(self, output, target):
    #     output = self.softmax(output)
    #     if not all([i == j for i, j in zip(output.shape, target.shape)]):
    #         target = self.onehot(target, output.shape[:2]+target.shape[2:])

    #     ratio = target.shape[-1] // output.shape[-1]
    #     if len(target.shape)==4:
    #         target = nn.AvgPool2d(kernel_size=ratio)(target)
    #     else:
    #         target = nn.AvgPool3d(kernel_size=ratio)(target)

    #     sum_axis = list(range(2,len(target.shape)))

    #     s = (10e-20)
    #     intersect = torch.sum(output * target,sum_axis)
    #     dice = (2 * intersect) / (torch.sum(output,sum_axis) + torch.sum(target,sum_axis) + s)
    #     #dice shape is (batch_size, nb_classes)
    #     return 1.0 - dice.mean()  
    def dice(self, output, target):
        output = self.softmax(output)

        # Downsample target BEFORE one-hot encoding
        if output.shape[2:] != target.shape[2:]:
            # Ensure target is class indices (B, 1, ...)
            if target.dim() == len(output.shape) and target.shape[1] == output.shape[1]:
                target = torch.argmax(target, dim=1, keepdim=True)
            elif target.dim() == len(output.shape) - 1:
                target = target.unsqueeze(1)

            target = F.interpolate(
                target.float(),
                size=output.shape[2:],
                mode='nearest'
            )

        # Now one-hot encode at the correct resolution
        target = self.onehot(target, output.shape)

        sum_axis = list(range(2, len(target.shape)))
        s = 1e-20
        intersect = torch.sum(output * target, sum_axis)
        dice = (2 * intersect) / (torch.sum(output, sum_axis) + torch.sum(target, sum_axis) + s)
        return 1.0 - dice.mean()


    def forward(self, output, target):
        result = self.dice(output, target)
        return result