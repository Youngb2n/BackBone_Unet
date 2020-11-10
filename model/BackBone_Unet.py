import torch.nn as nn
import torch
from model import DenseNet, EfficientNet, ResNet

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Bridge(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None):
        super().__init__()
        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels
        self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)
    
    def forward(self, up_x, down_x):
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x
    
    
    
#Resnet Backbone
class Res_Unet(nn.Module):
    DEPTH = 6
    def __init__(self,model, up_parm, n_classes = 1):
        super().__init__()
        resnet = model
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(up_parm[0], up_parm[0])

        up_blocks.append(UpBlock(up_parm[1]*2, up_parm[1], up_parm[0], up_parm[1]))
        up_blocks.append(UpBlock(up_parm[2]*2, up_parm[2], up_parm[1], up_parm[2]))
        up_blocks.append(UpBlock(up_parm[3]*2, up_parm[3], up_parm[2], up_parm[3]))
        up_blocks.append(UpBlock(in_channels=up_parm[4] + up_parm[5], out_channels=up_parm[4],
                                                    up_conv_in_channels=up_parm[3], up_conv_out_channels=up_parm[4]))
        up_blocks.append(UpBlock(in_channels=up_parm[6] + up_parm[7], out_channels=up_parm[6],
                                                    up_conv_in_channels=up_parm[4], up_conv_out_channels=up_parm[6]))
        self.up_blocks = nn.ModuleList(up_blocks)
        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)
    def forward(self,x):
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)
        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (Res_Unet.DEPTH - 1):
                continue    
            pre_pools[f"layer_{i}"] = x
        x = self.bridge(x)
        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{Res_Unet.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])
        x = self.out(x)

        return x

#densenetBackBone    
class Dense_Unet(nn.Module):
    DEPTH = 7
    def __init__(self, model, up_parm, n_classes = 1):
        super().__init__()
        densenet = model
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(densenet.children()))[:3]
        self.input_pool = list(densenet.children())[3]
        for bottleneck in list(densenet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(up_parm[0], up_parm[0])

        up_blocks.append(UpBlock(up_parm[1]*2, up_parm[1], up_parm[0], up_parm[1]))
        up_blocks.append(UpBlock(up_parm[2]*2, up_parm[2], up_parm[1], up_parm[2]))
        up_blocks.append(UpBlock(up_parm[3]*2, up_parm[3], up_parm[2], up_parm[3]))
        up_blocks.append(UpBlock(in_channels=up_parm[4] + up_parm[5], out_channels=up_parm[4],
                                                    up_conv_in_channels=up_parm[3], up_conv_out_channels=up_parm[4]))
        up_blocks.append(UpBlock(in_channels=up_parm[6] + up_parm[7], out_channels=up_parm[6],
                                                    up_conv_in_channels=up_parm[4], up_conv_out_channels=up_parm[6]))
        self.up_blocks = nn.ModuleList(up_blocks)
        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)
    def forward(self,x):
        pre_pools = dict()
        p =2 
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)
        for i, block in enumerate(self.down_blocks):            
            x = block(x)
            if (i %2 ==0) and (i != 6):
                pre_pools[f"layer_{p}"] = x
                p +=1
        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{Dense_Unet.DEPTH -2 - i}"
            x = block(x, pre_pools[key] )
        x = self.out(x)

        return x
    
#EfficentNet backbone

class Efficient_Unet(nn.Module):
    DEPTH = 7
    def __init__(self,model, up_parm, n_classes = 1):
        super().__init__()
        
        efficientNet = model
        down_blocks = []
        up_blocks = []
        for bottleneck in list(efficientNet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(up_parm[0], up_parm[0])

        up_blocks.append(UpBlock(up_parm[1]*2, up_parm[1], up_parm[0], up_parm[1]))
        up_blocks.append(UpBlock(up_parm[2]*2, up_parm[2], up_parm[1], up_parm[2]))
        up_blocks.append(UpBlock(up_parm[3]*2, up_parm[3], up_parm[2], up_parm[3]))
        up_blocks.append(UpBlock(in_channels=up_parm[4] + up_parm[5], out_channels=up_parm[4],
                                                    up_conv_in_channels=up_parm[3], up_conv_out_channels=up_parm[4]))
        up_blocks.append(UpBlock(in_channels=up_parm[6] + up_parm[7], out_channels=up_parm[6],
                                                    up_conv_in_channels=up_parm[4], up_conv_out_channels=up_parm[6]))
        self.up_blocks = nn.ModuleList(up_blocks)
        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)
    def forward(self,x):
        pre_pools = dict()
        p=0
        for i, block in enumerate(self.down_blocks):
            if i in (0,2,3,4,6):
                pre_pools[f"layer_{p}"] = x
                p+=1
            x = block(x)

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{Dense_Unet.DEPTH -2 - i}"
            x = block(x, pre_pools[key] )

        x = self.out(x)

        return x
    
    
def BackBone_Unet(backbone_name):
    up_parm_dict={
        'resnet18' : [512, 256, 128, 64, 64, 64, 64, 3],
        'resnet34' : [512, 256, 128, 64, 64, 64, 64, 3],
        'resnet50' : [2048, 1024, 512, 256, 128, 64, 64, 3],
        'resnet101' : [2048, 1024, 512, 256, 128, 64, 64, 3],
        'resnet152' : [2048, 1024, 512, 256, 128, 64, 64, 3],
        'densenet121' : [1024, 1024, 512, 256, 128, 64, 64, 3],
        'densenet161' : [2204, 2104, 752, 352, 128, 64, 64, 3],
        'densenet201' : [1920, 1792, 512, 256, 128, 64, 64, 3],
        'densenet169' : [1664, 1280, 512, 256, 128, 64, 64, 3],
        'efficientnet-b0' : [1280, 112, 40, 24, 16, 16, 64, 3],
        'efficientnet-b1' : [1280, 112, 40, 24, 16, 16, 64, 3],
        'efficientnet-b2' : [1280, 120, 48, 24, 16, 16, 64, 3],
        'efficientnet-b3' : [1280, 136, 48, 32, 24, 24, 64, 3],
        'efficientnet-b4' : [1280, 160, 56, 32, 24, 24, 64, 3],
        'efficientnet-b5' : [1280, 176, 64, 40, 24, 24, 64, 3],
        'efficientnet-b6' : [1280, 200, 72, 40, 32, 32, 64, 3],
        'efficientnet-b7' : [1280, 224, 80, 48, 32, 32, 64, 3]
        }

    efficient_param = {
        # 'efficientnet type': (width_coef, depth_coef, resolution, dropout_rate)
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 224, 0.2),
        'efficientnet-b2': (1.1, 1.2, 224, 0.3),
        'efficientnet-b3': (1.2, 1.4, 224, 0.3),
        'efficientnet-b4': (1.4, 1.8, 224, 0.4),
        'efficientnet-b5': (1.6, 2.2, 224, 0.4),
        'efficientnet-b6': (1.8, 2.6, 224, 0.5),
        'efficientnet-b7': (2.0, 3.1, 224, 0.5)
    }

    if backbone_name[0] =='r':
        if backbone_name[-2:] =='18':
            model = ResNet.ResNet18()
        if backbone_name[-2:] =='34':
            model = ResNet.ResNet34()
        if backbone_name[-2:] =='50':
            model = ResNet.ResNet50()
        if backbone_name[-2:] =='01':
            model = ResNet.ResNet101()
        if backbone_name[-2:] =='52':
            model = ResNet.ResNet152()
            
        net = Res_Unet(model = model, up_parm = up_parm_dict[backbone_name])

    elif backbone_name[0] =='d':
        if backbone_name[-2:] =='21': 
            model = DenseNet.DenseNet121(seon = False)
        if backbone_name[-2:] =='61':
            model = DenseNet.DenseNet161(seon = False)
        if backbone_name[-2:] =='01':
            model = DenseNet.DenseNet201(seon = False)   
        if backbone_name[-2:] =='69':
            model = DenseNet.DenseNet169(seon = False)

        net = Dense_Unet(model = model, up_parm = up_parm_dict[backbone_name])
    elif backbone_name[0] =='e':
        param = efficient_param[backbone_name]
        model = EfficientNet.EfficientNet(param)
        net = Efficient_Unet(model = model, up_parm = up_parm_dict[backbone_name])
        
    return net
