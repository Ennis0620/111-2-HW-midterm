import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data as Data
import os,math,shutil,tqdm
import numpy as np
from network import model_resnet50,model_resnet101,model_densenet121,model_efficientnet_v2_l,model_efficientnet_v2_s,\
model_resnext101_64x4d,model_densenet161,model_efficientnet_v2_l_trainable
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
from dataset import dataset
import pandas as pd

def test(test_loader,
            model,
        ):
    '''
    測試
    '''
    test_acc = 0
    total_size = 0
    total_correct = 0
    predict = np.array([])
    ans = np.array([])
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.eval()    #預測要把model變成eval狀態
    with torch.no_grad():
        for img  in tqdm.tqdm(test_loader):
            img = img.to(device)

            out = model(img)
            #累加每個batch的loss後續再除step數量
            m = nn.Softmax(dim=1)
            out = m(out)
            testid_p = out.argmax(dim=1)   
            
            #把所有預測和答案計算
            predict = np.append(predict,testid_p.cpu().numpy())
            
    return predict.astype(int)

class SquarePad:
	'''
	方形填充到長寬一樣大小再來resize
    '''
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return F.pad(image, padding, 0, 'constant')

if __name__ == '__main__':

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('GPU state:',device)
    CURRENT_PATH = os.path.dirname(__file__)
    BEST_WEIGHT_NAME = f'epoch_181_trainLoss_3.9698_trainAcc_95.13_valLoss_1.319_valAcc_99.06.pth'
    TMP_ROOT = f'model_weight/model_efficientnet_v2_l'
    SAVE_MODELS_PATH = f'{CURRENT_PATH}/{TMP_ROOT}/{BEST_WEIGHT_NAME}'
    model = model_efficientnet_v2_l(num_classes=6)
    model.load_state_dict(torch.load(SAVE_MODELS_PATH))
    #print(model)
    print('Loading Weight:',BEST_WEIGHT_NAME)

    NEURAL_NETWORK = model.to(device) 
    
    SPLIT_RATIO = 0.2
    SHUFFLE_DATASET = True
    BATCH_SIZE=128

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    # test dataset
    test_dataset = dataset(type = 'test',
                            transform = test_transform
                            )
    # test data loaders
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False
    )

    predict= test(test_loader=test_loader,
                model=NEURAL_NETWORK,)
    
    print(predict)

    test_csv = f'{CURRENT_PATH}/test.csv'
    test_csv = pd.read_csv(test_csv)
    test_csv['Label'] = predict
    test_csv.to_csv('submmit.csv',index=False)
    #print(test_csv.info())
    
         

    
    