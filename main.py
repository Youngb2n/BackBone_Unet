from model import BackBone_Unet
import argparse
import os
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from losses import dice_loss,calc_loss,print_metrics,dice_coeff
import torch.nn as nn
import torch.optim as optim
import data
from collections import defaultdict
import torch.nn.functional as F
import torch
import copy
import utils
from collections import defaultdict
parser = argparse.ArgumentParser(description='Learn by Modeling Pathology DataSet')
parser.add_argument('--backbone',type=str,default= 'resnet18' , help='Select backbone model name')
parser.add_argument('--lr',type=float,default= 0.001 , help='Select opimizer learning rate')
parser.add_argument('--epochs',type=int,default= 30 , help='Select train epochs')
parser.add_argument('--data_path', type=str, default='/content/gdrive/My Drive/membrane',
                    help=' Check Dataset directory.')
parser.add_argument('--save_path', type=str, default='/workspace/data/Pytorch/weights/',
                    help=' Check Save Path directory.')

args = parser.parse_args()
PATH = args.data_path
train_path = os.path.join(PATH, 'train')
val_path  = os.path.join(PATH, 'test')

trans = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
mask_trans = transforms.Compose([
    transforms.Resize(224),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

train_set = data.SegDataset(train_path, trans, mask_trans)
val_set = data.SegDataset(val_path ,trans, mask_trans)

image_datasets = {
    'train': train_set,
     'val': val_set
}
batch_size = 1

dataloaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
}
history = {'train' :{'dice': [],
            'loss': [],
            'bce': []},
            'val':{'dice': [],
            'loss': [],
            'bce': []}}
    
model = BackBone_Unet.BackBone_Unet(args.backbone)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

if torch.cuda.is_available():
    torch.cuda.synchronize()

def train_model(model,dataloaders, optimizer, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase, history)
            epoch_loss = metrics['loss'] / epoch_samples


            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    print('Best val loss: {:4f}'.format(best_loss))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
model = train_model(model, dataloaders , optimizer, num_epochs=args.epochs)
utils.train_graph(args.epochs,history, 'BackBone_Unet')
utils.test(model, dataloaders)
