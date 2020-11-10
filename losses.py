import torch.nn.functional as F
import torch

# def dice_loss(pred, target):
#     """This definition generalize to real valued pred and target vector.
# This should be differentiable.
#     pred: tensor with first dimension as batch
#     target: tensor with first dimension as batch
#     """

#     smooth = 1.

#     # have to use contiguous since they may from a torch.view op
#     iflat = pred.contiguous().view(-1)
#     tflat = target.contiguous().view(-1)
#     intersection = (iflat * tflat).sum()

#     A_sum = torch.sum(iflat * iflat) 
#     B_sum = torch.sum(tflat * tflat)
    
#     return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)
    
    loss = bce * bce_weight + dice * (1 - bce_weight) # bce + dice

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    
    del bce
    del pred
    del dice    
    return loss

def dice_coeff(pred, target):
    return 1- dice_loss(pred, target, smooth = 1.)


def print_metrics(metrics, epoch_samples, phase, history):    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        history[phase][k].append(round(metrics[k] / epoch_samples,3))
    print("{}: {}".format(phase, ", ".join(outputs)))   
