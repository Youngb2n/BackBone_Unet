from torch.utils.data import Dataset
import os
from PIL import Image
class SegDataset(Dataset):
    def __init__(self, train_path, transform=None,mask_trans=None):
        self.Train_path = train_image_path = os.path.join(train_path,'image')
        self.Label_path = train_image_path = os.path.join(train_path,'label')
        self.Image_List = os.listdir(self.Train_path)
        self.mask_List =  os.listdir(self.Label_path)
        self.transform = transform
        self.mask_trans = mask_trans
    def __len__(self):
        return len(self.Image_List)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.Train_path,self.Image_List[idx])).convert('RGB')
        mask = Image.open(os.path.join(self.Label_path,self.mask_List[idx])).convert('RGB')
        if self.transform:
            img = self.transform(img)
            mask = self.mask_trans(mask)
        mask = (mask>0.5).float()
        return [img, mask]
