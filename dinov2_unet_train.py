import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from loss import comb_loss, acc
from dataset_load import load_dataset

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


def train_model(data, epochs, device, classes):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
        
    Unet = unet(in_chan = 3, out_chan = 10).to(device)
    decoder = Decoder(feature_dims = 384).to(device)
    
    optimizer = optim.Adam(list(Unet.parameters())+list(decoder.parameters()), lr = 1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.75, patience = 5, verbose = 1, threshold = 1e-3, cooldown = 0, min_lr = 1e-7)
    
    tl,vl,ta,va = [],[],[],[]
    val_best = float('inf')
    
    for epoch in range(epochs):
        print(f'Epoch: {epoch+1}')
        Unet.train()
        decoder.train()
        tlt,tat,vlt,vat = 0.0,0.0,0.0,0.0
        for i, (images, masks) in enumerate(data['train']):
            
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                features = model.forward_features(images)['x_norm_patchtokens']
                features = features.reshape(images.size(0), 32, 32, features.size(2)).permute(0, 3, 1,2)
                features = features.to(device)
            features = decoder(features)
            logits = Unet(images, features)
            loss = comb_loss(logits, masks)
            accur = acc(logits, masks)
            loss.backward()
            optimizer.step()
            loss_batch = loss.item()
            acc_batch = accur.item()
            tlt += loss_batch
            tat += acc_batch
            print(f'Batch No.: {i+1}, Train Loss: {loss_batch:.4f}, Train Acc: {acc_batch*100:.2f}')
        i = i+1
        
        tl_avg = tlt/i
        ta_avg = tat/i
        tl.append(tl_avg)
        ta.append(ta_avg)
        
        Unet.eval()
        decoder.eval()
        with torch.no_grad():
            for i, (images, masks) in enumerate(data['val']):
                
                images = images.to(device)
                masks = masks.to(device)
                features = model.forward_features(images)['x_norm_patchtokens']
                features = features.reshape(images.size(0), 32, 32, features.size(2)).permute(0, 3, 1,2)
                features = features.to(device)
                features = decoder(features)
                logits = Unet(images, features)
                
                loss = comb_loss(logits, masks)
                accur = acc(logits, masks)
                loss_val = loss.item()
                acc_val = accur.item()
                vlt += loss_val
                vat += acc_val
                print(f'Val Batch: {i+1}, Val Loss: {loss_val:.4f}, Val Acc: {acc_val*100:.2f}')
            i = i+1
            vl_avg = vlt/i
            va_avg = vat/i
            vl.append(vl_avg)
            va.append(va_avg)
            scheduler.step(vl_avg)
            
            if vl_avg<val_best:
                print(f'val loss came down from {val_best} to {vl_avg}')
                val_best = vl_avg
                torch.save(Unet.state_dict(), 'saved_models/DINOv2_unet.pth')
                torch.save(decoder.state_dict(), 'saved_models/DINOv2_unetdec.pth')
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {tl_avg:.4f}, Train Acc: {ta_avg*100:.2f}, Val Loss {vl_avg:.4f}, Val Acc: {va_avg*100:.2f}')
    return tl, ta, vl, va

def loss_acc(tl, vl, ta, va):
    e = range(1, len(tl)+1)
    plt.figure(figsize = (10, 6))

    plt.subplot(1, 2, 1)
    plt.plot(e, ta, label = 'Train Accuracy')
    plt.plot(e, va, label = 'Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc = 'upper left')
    
    plt.subplot(1,2,2)
    plt.plot(e, tl, label = 'Train Loss')
    plt.plot(e, vl, label = 'Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc = 'upper right')

    plt.tight_layout()
    plt.savefig(os.path.join('Figures/DINOv2_unet', 'loss_acc.png'), bbox_inches = 'tight')
    plt.show()

path = 'dataset'
subfolders = ['train', 'val']
h = 448
w = 448
classes = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

data = load_dataset(path, subfolders, h, w, classes)

epochs = 70

tl, ta, vl, va = train_model(data, epochs, device, classes)

loss_acc(tl, vl, ta, va) 
