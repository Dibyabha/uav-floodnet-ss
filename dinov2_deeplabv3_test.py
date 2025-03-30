import numpy as np
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.metrics import jaccard_score
from dataset_load import load_dataset
from loss import comb_loss, acc

class Decoder(nn.Module):
    def __init__(self, feature_dims):
        super(Decoder, self).__init__()
        self.conv = nn.Conv2d(feature_dims, 256, kernel_size = 1, stride = 1) 
        self.upsample = nn.Upsample(size = (56, 56), mode = 'bicubic', align_corners = False) 
        self.conv_af_concat = nn.Conv2d(512, 256, kernel_size = 1, stride = 1)
        
    def forward(self, xdino, xdl):
        xdino = self.conv(xdino)
        xdino = self.upsample(xdino)
        concat = torch.cat([xdino, xdl], dim = 1)
        out = self.conv_af_concat(concat)
        return out

def eval_model(data, model, deeplab, decoder, device, classes):
    
    model.eval()
    decoder.eval()
    deeplab.eval()
    test_loss = 0
    test_acc = 0
    all_true = []
    all_pred = []
    with torch.no_grad():
        for i, (images, masks) in enumerate(data['test']):
            images = images.to(device)
            masks = masks.to(device)
            features = model.forward_features(images)['x_norm_patchtokens']
            features = features.reshape(images.size(0), 32, 32, features.size(2)).permute(0, 3, 1,2)
            features = features.to(device)
            backbone = deeplab.backbone(images)
            aspp = deeplab.classifier[0](backbone['out']).to(device)
            
            out = decoder(features, aspp)
            for j in range(1, len(deeplab.classifier)):
                out = deeplab.classifier[j](out).to(device)

            prob = F.interpolate(out, size = (448, 448), mode = 'bicubic', align_corners = False)
            prob = F.softmax(prob, dim = 1)
            loss = comb_loss(prob, masks)
            accur = acc(prob, masks)
            loss_test = loss.item()
            acc_test = accur.item()
            test_loss += loss_test
            test_acc += acc_test
            print(f'Test Batch: {i+1}, Test Loss: {loss_test:.4f}, Test Acc: {acc_test*100:.2f}')
            for k in range(images.size(0)):
                
                true = torch.argmax(masks[k], dim = 0).cpu().numpy()
                pred = torch.argmax(prob[k], dim = 0).cpu().numpy()
                all_true.append(true)
                all_pred.append(pred)
        i = i+1
        tl_avg = test_loss/i
        ta_avg = test_acc/i
    all_true = np.stack(all_true)
    all_pred = np.stack(all_pred)
    miou, iou = calc_miou(all_true, all_pred, classes)
    print(f'mIoU:{miou:.4f}')
    print(iou)
    return tl_avg, ta_avg

def plot(data, model, deeplab, decoder, classes, figures_path, device):
    model.eval()
    decoder.eval()
    deeplab.eval()
    cmap = plt.cm.get_cmap('tab10', classes)
    num = 200
    test = data['test']
    indices = np.linspace(0, len(test.dataset)-1, num, dtype = int)
    
    for i in indices:
        image, mask = test.dataset[i]
        image = image.unsqueeze(0).to(device)
        mask = mask.to(device)
        

        features = model.forward_features(image)['x_norm_patchtokens']
        features = features.reshape(image.size(0), 32, 32, features.size(2)).permute(0, 3, 1, 2)
        features = features.to(device)
        backbone = deeplab.backbone(image)
        aspp = deeplab.classifier[0](backbone['out']).to(device)
    
        out = decoder(features, aspp)
        for j in range(1, len(deeplab.classifier)):
            out = deeplab.classifier[j](out).to(device)

        prob = F.interpolate(out, size = (448, 448), mode = 'bicubic', align_corners = False)
        prob = F.softmax(prob, dim = 1)
        pred = torch.argmax(prob, dim = 1).cpu().numpy().squeeze()
        true = torch.argmax(mask, dim = 0).cpu().numpy()

        unique_true_classes = np.unique(true)
        unique_pred_classes = np.unique(pred)

        print(f"Image {i} - Unique classes in Actual Mask: {unique_true_classes}")
        print(f"Image {i} - Unique classes in Predicted Mask: {unique_pred_classes}")

        norm = Normalize(vmin = 0, vmax = classes-1)
        
        plt.figure(figsize = (25, 15))

        ax1 = plt.subplot(1, 3, 1)
        ax1.imshow(image.squeeze().cpu().numpy().transpose(1, 2, 0))
        ax1.set_title('Image')
        ax1.axis('off')

        ax2 = plt.subplot(1, 3, 2)
        im2 = ax2.imshow(true, cmap = cmap, norm = norm)
        ax2.set_title('Actual Mask')
        ax2.axis('off')
        cbar2 = plt.colorbar(im2, ax = ax2, ticks = range(classes), fraction = 0.046, pad = 0.04)
        cbar2.ax.set_ylabel('Class', rotation = 270, labelpad = 15)
        cbar2.ax.tick_params(labelsize = 8)

        ax3 = plt.subplot(1, 3, 3)
        im3 = ax3.imshow(pred, cmap = cmap, norm = norm)
        ax3.set_title('Predicted Mask')
        ax3.axis('off')
        cbar3 = plt.colorbar(im3, ax = ax3, ticks = range(classes), fraction = 0.046, pad = 0.04)
        cbar3.ax.set_ylabel('Class', rotation = 270, labelpad = 15)
        cbar3.ax.tick_params(labelsize = 8)

        plt.tight_layout()
        plt.savefig(os.path.join(figures_path, f'prediction_{i}.png'), bbox_inches = 'tight')
        plt.show()


path = 'dataset'
subfolders = ['test']
h = 448
w = 448
classes = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

data = load_dataset(path, subfolders, h, w, classes)

model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model = model.to(device)
model.to(device)

decoder = Decoder(feature_dims = 384).to(device)
decoder.load_state_dict(torch.load('saved_models/DINOv2_deepdec.pth'))

deeplab = deeplabv3_resnet50(weights = None).to(device)
deeplab.classifier[4] = nn.Conv2d(256, classes, kernel_size = 1, stride = 1).to(device)
deeplab.load_state_dict(torch.load('saved_models/DINOv2_deep.pth'),strict = False)

loss, acc = eval_model(data, model, deeplab, decoder, device, classes)

print(f'Test Loss: {loss:.4f}') 
print(f'Test Acc: {acc*100:.2f}')

plot(data, model, deeplab, decoder, classes, 'Figures/DINOv2_deeplabv3', device)
