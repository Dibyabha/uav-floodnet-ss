import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.metrics import jaccard_score
from dataset_load import load_dataset
from loss import comb_loss, acc

def calc_miou(y_true, y_pred, classes):
    iou_per_class = []
    for i in range(1, classes):
        true_class = (y_true == i).astype(int).flatten()
        pred_class = (y_pred == i).astype(int).flatten()
        iou = jaccard_score(true_class, pred_class, average='binary')
        iou_per_class.append(iou)
    mean_iou = np.mean(iou_per_class)
    return mean_iou, iou_per_class

class Decoder(nn.Module):
    def __init__(self, feature_dims = 384):
        super(Decoder, self).__init__()
        self.conv = nn.Conv2d(feature_dims, 512, kernel_size = 1, padding = 1)
        self.upsample = nn.Upsample(size = (28, 28), mode = 'bicubic', align_corners = False)
        
    def forward(self, features):
        feat = self.conv(features)
        feat = self.upsample(feat)
        
        return feat
    
class conv_block(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(conv_block, self).__init__()
        self.conv_layers = nn.Sequential(nn.Conv2d(in_chan, out_chan, kernel_size = 3, padding = 1), nn.BatchNorm2d(out_chan), nn.ReLU(inplace=True),
                                 nn.Conv2d(out_chan, out_chan, kernel_size = 3, padding = 1), nn.BatchNorm2d(out_chan), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv_layers(x)

class up(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_chan, out_chan, kernel_size = 4, stride = 2, padding = 1)
        self.conv = conv_block(in_chan, out_chan)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim = 1)
        return self.conv(x)
        
class unet(nn.Module):
    def __init__(self, in_chan = 3, out_chan = 10):
        super(unet, self).__init__()
        self.enc1 = conv_block(in_chan, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        self.bottle = conv_block(1024, 1024)
        
        self.dec4 = up(1024, 512)
        self.dec3 = up(512, 256)
        self.dec2 = up(256, 128)
        self.dec1 = up(128, 64)
        self.dec0 = conv_block(64, out_chan)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

    def forward(self, xunet, xdino):
        e1 = self.enc1(xunet)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        e4 = self.enc4(p3)
        p4 = self.pool(e4)
        
        x = self.bottle(torch.cat([p4, xdino], dim = 1))
        
        d4 = self.dec4(x, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        d0 = self.dec0(d1)
        out = F.softmax(d0, dim = 1)
        
        return out

def eval_model(data, model, decoder, Unet, device, classes):
    
    model.eval()
    decoder.eval()
    Unet.eval()
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
            features = decoder(features)
            logits = Unet(images, features)
            
            loss = comb_loss(logits, masks)
            accur = acc(logits, masks)
            loss_test = loss.item()
            acc_test = accur.item()
            test_loss += loss_test
            test_acc += acc_test
            print(f'Test Batch: {i+1}, Test Loss: {loss_test:.4f}, Test Acc: {acc_test*100:.2f}')
            for j in range(images.size(0)):
                
                true = torch.argmax(masks[j], dim = 0).cpu().numpy()
                pred = torch.argmax(logits[j], dim = 0).cpu().numpy()
                
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

def plot(data, model, decoder, Unet, classes, figures_path, device):
    model.eval()
    decoder.eval()
    Unet.eval()
    cmap = plt.cm.get_cmap('tab10', classes)
    num = 200
    test = data['test']
    indices = np.linspace(0, len(test.dataset)-1, num, dtype = int)
    
    for i in indices:
        image, mask = test.dataset[i]
        image = image.unsqueeze(0).to(device)
        mask = mask.to(device)
        features = model.forward_features(image)['x_norm_patchtokens']
        features = features.reshape(image.size(0), 32, 32, features.size(2)).permute(0, 3, 1,2)
        features = features.to(device)
        features = decoder(features)
        logits = Unet(image, features)
        pred = torch.argmax(logits, dim = 1).cpu().numpy().squeeze()
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
        plt.savefig(os.path.join(figures_path, f'prediction_{i}_changed.png'), bbox_inches = 'tight')
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

Unet = unet(in_chan = 3, out_chan = 10).to(device)
Unet.load_state_dict(torch.load('saved_models/DINOv2_unet.pth'))

decoder = Decoder(feature_dims = 384).to(device)
decoder.load_state_dict(torch.load('saved_models/DINOv2_unetdec.pth'))

loss, acc = eval_model(data, model, decoder,Unet, device, classes)

print(f'Test Loss: {loss:.4f}') 
print(f'Test Acc: {acc*100:.2f}')

plot(data, model, decoder, Unet, classes, 'Figures/DINOv2_unet', device)
