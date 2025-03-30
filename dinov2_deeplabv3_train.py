import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
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
        
def train_model(data, epochs, device, classes):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
        
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    deeplab = deeplabv3_resnet50(weights = weights).to(device)
    deeplab.classifier[4] = nn.Conv2d(256, classes, kernel_size = 1, stride = 1).to(device)
    
    for param in deeplab.backbone.parameters():
        param.requires_grad = False
    
    decoder = Decoder(feature_dims = 384).to(device)
    
    optimizer = optim.Adam(list(deeplab.classifier.parameters()) + list(decoder.parameters()), lr = 1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.75, patience = 5, verbose = 1, threshold = 1e-3, cooldown = 0, min_lr = 1e-7)
    
    tl,vl,ta,va = [],[],[],[]
    val_best = float('inf')
    for epoch in range(epochs):
        print(f'Epoch: {epoch+1}')
        deeplab.train()
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
            
            backbone = deeplab.backbone(images)
            aspp = deeplab.classifier[0](backbone['out']).to(device)
            
            out = decoder(features, aspp)
            for j in range(1, len(deeplab.classifier)):
                out = deeplab.classifier[j](out).to(device)
            
            prob = F.interpolate(out, size = (448, 448), mode = 'bicubic', align_corners = False)
            prob = F.softmax(prob, dim = 1)
            loss = comb_loss(prob, masks)
            accur = acc(prob, masks)
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
        
        decoder.eval()
        deeplab.eval()
        with torch.no_grad():
            for i, (images, masks) in enumerate(data['val']):
                
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
                torch.save(decoder.state_dict(), 'saved_models/DINOv2_deepdec.pth')
                torch.save(deeplab.state_dict(), 'saved_models/DINOv2_deep.pth')
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {tl_avg:.4f}, Train Acc: {ta_avg:.4f}, Val Loss {vl_avg:.4f}, Val Acc: {va_avg:.4f}')
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
    plt.savefig(os.path.join('Figures/DINOv2_deeplabv3', 'loss_acc.png'), bbox_inches = 'tight')
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
