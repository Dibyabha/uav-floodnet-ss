import torch

def dice_loss(pred, true, smooth = 1e-5):
    
    pred_flat = pred.view(pred.size(0), pred.size(1), -1)
    true_flat = true.view(true.size(0), true.size(1), -1)
    inter = torch.sum(true_flat*pred_flat, dim = 2)
    union = torch.sum(true_flat, dim = 2)+torch.sum(pred_flat, dim = 2)
    dice = (2. * inter+smooth)/(union+smooth)
    dl = torch.mean(1-dice, dim = 1)
    return dl

def focal_loss(pred, true, gamma = 4.0, alpha = None):
    if alpha is None:
        alpha = torch.tensor([0.08, 0.14, 0.10, 0.10, 0.05, 0.015, 0.01, 0.25, 0.25, 0.005], dtype = torch.float32)
    
    alpha_res = alpha.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    alpha_res = alpha_res.permute(0, 3, 1, 2)
    alpha_res = alpha_res.to(device='cuda')
    alpha_weights = alpha_res*true
    alpha_weights = alpha_weights.to(device='cuda')
    pred_clip = torch.clip(pred, 1e-7, 1-1e-7)
    cross_ent = -true*torch.log(pred_clip)
    loss = alpha_weights*(1-pred_clip)**gamma*cross_ent
    fl = torch.mean(torch.sum(loss, dim = [2, 3]), dim = 1)
    return fl

def comb_loss(pred, true):
    dl = dice_loss(pred, true)
    fl = focal_loss(pred, true)
    return 0.5*torch.mean(dl)+0.5*torch.mean(fl)
    
def acc(pred, true):
    pred_classes = torch.argmax(pred, dim = 1)
    true_classes = torch.argmax(true, dim = 1)
    correct = (pred_classes == true_classes).sum().float()
    total = pred.size(0)*pred.size(2)*pred.size(3)
    return correct/total
