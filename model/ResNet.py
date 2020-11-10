import torch
import torch.nn as nn
import torch.nn.functional as F

class _BasicBlock(nn.Module):
    expansion = 1
    def __init__(self,inplanes,planes,stride =1,seon = False):
        super(_BasicBlock,self).__init__()
        self.seon= seon

        self.conv1 = nn.Conv2d(inplanes,planes,kernel_size=3,stride=stride, padding = 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)       
        self.relu = nn.ReLU(inplace =True)

        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.se = SELayer(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion*planes,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        
    def forward(self,x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        if self.seon:
            out = self.se(out)
        return out

class _Bottleneck(nn.Module):
    expansion = 4
    def __init__(self,inplanes,planes,stride=1,seon = False):
        super(_Bottleneck,self).__init__()
        self.seon =seon

        self.conv1 = nn.Conv2d(inplanes,planes,kernel_size=1,stride=1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes,kernel_size=1,stride=1,bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion*planes,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)

        if self.seon:
            out = self.se(out)
        out = self.relu(out)
        return out

class SELayer(nn.Module):
    def __init__(self, planes, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Linear(planes, planes // reduction, bias=False)
        self.relu =  nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(planes // reduction, planes, bias=False)
        self.fc = nn.Sequential(
            nn.Linear(planes, planes // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(planes // reduction, planes, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        out = self.avg_pool(x)
        out = out.view(b, c)
        out = self.fc(out)
        out = out.view(b, c, 1, 1)
        return x * out.expand_as(x)


class ResNet(nn.Module):
    def __init__(self,block,num_blocks,num_classes=1, seon = False):
        super(ResNet, self).__init__()
        self.seon = seon
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,stride=2,padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.block1 = self._make_layer(block,num_blocks[0],64,stride=1)
        self.block2 = self._make_layer(block,num_blocks[1],128,stride=2)
        self.block3 = self._make_layer(block,num_blocks[2],256,stride=2)
        self.block4 = self._make_layer(block,num_blocks[3],512,stride=2)
        self.linear = nn.Linear(512*block.expansion,num_classes)
        

    def _make_layer(self,block,num_blocks,planes,stride):
        layers = []
        layers+=[block(self.inplanes,planes,stride=stride,seon= self.seon)]
        self.inplanes = planes*block.expansion
        for i in range(1,num_blocks):
            layers+=[block(self.inplanes,planes,seon =self.seon)]
            self.inplanes = planes*block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = nn.AdaptiveAvgPool2d(1)(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
def ResNet18(seon=False):
    return ResNet(_BasicBlock, [2,2,2,2],num_classes=2,seon=seon)

def ResNet34(seon=False):
    return ResNet(_BasicBlock, [3,4,6,3],num_classes=2, seon=seon)

def ResNet50(seon=False):
    return ResNet(_Bottleneck, [3,4,6,3], num_classes=2, seon=seon)
def ResNet101(seon=False):
    return ResNet(_Bottleneck, [3,4,23,3],num_classes=2, seon=seon)

def ResNet152(seon=False):
    return ResNet(_Bottleneck, [3,8,36,3],num_classes=2, seon=seon)
