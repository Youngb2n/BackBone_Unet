import os
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
def train_graph(epoch,history_dict,model_name):
    # makefolder
    try:
        if not os.path.exists('/content/segsample/result'):
            os.makedirs('/content/segsample/results')
    except OSError:
        print('Error Creating director')
    
    epochs_range = range(1,epoch+1)
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, history_dict['train']['bce'],color='r' ,label='Traing BCE')
    plt.plot(epochs_range, history_dict['val']['bce'], color='b', label='Validation BCE')
    plt.title('Training and Validation BCE')

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, history_dict['train']['dice'], color='r',label='Training Dice Loss')
    plt.plot(epochs_range, history_dict['val']['dice'], color='b', label='Validation Dice Loss')
    plt.title('Training and Validation Dice Loss')

    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, history_dict['train']['loss'],color='r', label='Training Loss')
    plt.plot(epochs_range, history_dict['val']['loss'], color='b', label='Validation Loss')
    plt.title('Training and Validation Loss')

    plt.savefig('/content/segsample/results/Segmentation_{}train_val_graph.png' .format(model_name), dpi=80)
    
    print('')
    print('The train graph is saved...')
    print('')

def test(model, dataloaders):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad(): 

        loss_arr = []
        epoch_acc = []
        
        for batch, (img,mask) in enumerate(dataloaders['val']):
            print(batch)
            model.eval() # 네트워크를 evaluation 용으로 선언

            x_tensor = img.to(device)
            y_tensor = mask.to(device)
            prediction = model(x_tensor)
            pr_mask = prediction.squeeze(0)
            pr_mask = torch.transpose(pr_mask, 2,0)
            pr_mask = pr_mask.data.cpu().numpy()


            bi_mask = model(x_tensor).round()
            bi_mask = bi_mask.squeeze(0)
            bi_mask = bi_mask.permute(2,1,0)

            img= img.squeeze()
            img = img.permute(2,1,0)
            #     img = cv2.cvtColor(img.cpu().numpy(), cv2.COLOR_BGR2RGB)

            y_tensor = y_tensor.squeeze(3)
            y_tensor = torch.transpose(y_tensor , 2,0)
            gt_mask = cv2.cvtColor(y_tensor.cpu().numpy(), cv2.COLOR_GRAY2RGB)
            pr_mask = cv2.cvtColor(pr_mask, cv2.COLOR_GRAY2RGB)

            mask = mask.squeeze(0)
            mask = mask.permute(2,1,0)
            mask = mask.data.cpu().numpy()

            gt_mask = mask.astype(np.float32)
            gt_mask = gt_mask * [1,1,1]
            pr_mask = pr_mask * [1,1,1]

            bi_mask = bi_mask.squeeze()
            bi_mask = bi_mask.data.cpu().numpy()
            bi_mask = bi_mask * [255]
#             bi_mask = np.squeeze(bi_mask,axis=1)

            plt.figure(figsize=(20, 10))

            plt.subplot(1, 4, 1)
            plt.xlabel("original ")
            plt.imshow(img) # original Image

            plt.subplot(1, 4, 2)
            plt.xlabel("GT ")
            plt.imshow(gt_mask) # GT mask

            plt.subplot(1, 4, 3)
            plt.xlabel("Prediction ")
            plt.imshow(pr_mask) # prediction mask

            plt.subplot(1, 4, 4)
            plt.xlabel("Binary ")
            plt.imshow(bi_mask, cmap='binary') # prediction mask

            plt.savefig(f'/content/segsample/results/predict_img{batch}.png', dpi=80)
