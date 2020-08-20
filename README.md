# BackBone Unet

실험주제
-------
Segmentaion task의 Unet Model 구조에서의 블럭을 classification 모델구조의 ResNet, DenseNet, EfficientNet의 블럭을 BackBone으로 하여 실험을 진행함.
그에따른 모델의 성능향상이 있는지 알아보고자 함.   

실험환경
--------
* Google Colab Gpu 을 사용하였음.
* Colab 내장 라이브러리 PyTorch '1.5.0' 환경    

DataSet
-------
<http://brainiac2.mit.edu/isbi_challenge> 참조

Training
--------
```
! python3 /content/segsample/main.py -h
```
```
usage: main.py [-h] [--backbone BACKBONE] [--lr LR] [--epochs EPOCHS]
               [--data_path DATA_PATH] [--save_path SAVE_PATH]

Learn by Modeling Pathology DataSet

optional arguments:
  -h, --help            show this help message and exit
  --backbone BACKBONE   Select backbone model name
  --lr LR               Select opimizer learning rate
  --epochs EPOCHS       Select train epochs
  --data_path DATA_PATH
                        Check Dataset directory.
  --save_path SAVE_PATH
                        Check Save Path directory.
```

Result
------
> DataSet의 데이터 수가 적어 모델의 성능을 비교하기에 적절하지 않았음.
> 추후에 다른 데이터세트로 모델의 성능을 비교할 예정.    

### ResNet18   
<img src="/image/1.png" width="80%" height="80%" title="img2" alt="img2"></img>   
### ResNet50   
<img src="/image/2.png" width="80%" height="80%" title="img2" alt="img2"></img>   

### DenseNet121   
<img src="/image/3.png" width="80%" height="80%" title="img2" alt="img2"></img>   

### EfficientNet-B0   
<img src="/image/4.png" width="80%" height="80%" title="img2" alt="img2"></img>   
