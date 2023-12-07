import torch
import torchvision
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
#//==0.flask==//
import io
import json
from flask import Flask, jsonify, request
from io import BytesIO
#//====//


# 이미지를 불러오기
#image = Image.open("lectuce_healthy_1.jpg")


def accuracy(outputs, labels):
  _, preds = torch.max(outputs, dim=1)
  return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):

  def training_step(self,batch):
    images,labels = batch
    out = self(images)
    loss = F.cross_entropy(out,labels)
    return loss

  def validation_step(self,batch):
    images,labels = batch
    out = self(images)
    loss = F.cross_entropy(out,labels)
    acc = accuracy(out,labels)
    return {'val_loss':loss,'val_acc':acc}

  def validation_epoch_end(self,outputs):
    batch_loss = [out['val_loss'] for out in outputs]
    epoch_loss = torch.stack(batch_loss).mean()
    batch_acc = [out['val_acc'] for out in outputs]
    epoch_acc = torch.stack(batch_acc).mean()
    return {'val_loss':epoch_loss.item(),'val_acc':epoch_acc.item()}

  def epoch_end(self,epoch,result):
    print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['train_loss'], result['val_loss'], result['val_acc']))

class Plant_Disease_Model2(ImageClassificationBase):

  def __init__(self):
    super().__init__()
    self.network = models.resnet34(weights=None)

    num_ftrs = self.network.fc.in_features
    self.network.fc = nn.Linear(num_ftrs,28)

  def forward(self,xb):
    out = self.network(xb)
    return out

class Plant_Disease_Model(ImageClassificationBase):
  
  def __init__(self):
    super().__init__()
    self.network = nn.Sequential(
        nn.Conv2d(3,32,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2), #output : 64*64*64

        nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2), #output : 128*32*32

        nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2), #output : 256*16*16
        
        nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2), #output : 512*8*8
        
        nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.Conv2d(512,1024,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2), #output : 1024*4*4
        nn.AdaptiveAvgPool2d(1),
        
        nn.Flatten(),
        nn.Linear(1024,512),
        nn.ReLU(),
        nn.Linear(512,256),
        nn.ReLU(),
        nn.Linear(256,2)
        )
    
  def forward(self,xb):
    out = self.network(xb)
    return out
  

#//==1.flask ==//
app = Flask(__name__)
#//==flask==//


classes = ['bean__bean_spot', 'bean__blight', 'bean__brown_spot', 'bean__healthy', 'corn__common_rust', 'corn__gray_spot', 'corn__healthy', 'green_onion__black_spot', 'green_onion__downy_mildew', 'green_onion__healthy', 'green_onion__rust', 'lectuce__downy_mildew', 'lectuce__drop', 'lectuce__healthy', 'pepper__anthracnose', 'pepper__healthy', 'pepper__powdery_mildew', 'potato__Early_Blight', 'potato__healthy', 'potato__late_Blight', 'potato__soft_rot', 'pumpkin__healthy', 'pumpkin__leaf_mold', 'pumpkin__mosaic', 'pumpkin__powdery_mildew', 'radish__black_spot', 'radish__downy_mildew', 'radish__healthy']

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        #img_bytes = file.read()

        # 이미지를 전처리
        transform = transforms.Compose([
        transforms.Resize(size=128),
        transforms.ToTensor(),
        ])

        image = Image.open(BytesIO(file.read()))
        image = transform(image)
        image = torch.unsqueeze(image, 0)

        # 모델 로드
        state_dict1 = torch.load("./models/s1_binary_class.pth", map_location=torch.device('cpu'))
        state_dict2 = torch.load("./models/s3_disease_class.pth", map_location=torch.device('cpu'))

        # 모델 불러오고 evaluation 상태로 지정
        model1 = Plant_Disease_Model()
        model2 = Plant_Disease_Model2()

        model1.load_state_dict(state_dict1)
        model1.eval()

        model2.load_state_dict(state_dict2)
        model2.eval()

        # 추론을 수행
        output1 = model1(image)
        pred_class = output1.argmax()

        if pred_class.item() == 0: # 식물 아닌 경우print("식물 아님")
            print("식물아님")

            top1_class, top2_class = "식물아님"
            probs = [100, 100]
            print(top1_class, top2_class)

        if pred_class.item() == 1: # 식물인경우
           print("식물임")
           output2 = model2(image)
           _, preds = torch.topk(output2, 2)
           probs = torch.nn.functional.softmax(output2, dim=1)[0,  preds[0]]
           print(preds, probs)

           top1_class = classes[preds[0][0]]
           top2_class = classes[preds[0][1]]

        print(top1_class, top2_class)
        # Tensor를 Python의 기본 자료형으로 변환
        top1_class = top1_class.item()
        top2_class = top2_class.item()
        top1_percent = probs[0].item()
        top2_percent = probs[1].item()

        # JSON으로 응답
        return jsonify({"top1_class": top1_class, "top2_class": top2_class, "top1_percent": top1_percent, "top2_percent": top2_percent})




        # # 가장 높은 확률을 가진 클래스를 예측합니다.
        # pred_class = output.argmax()
        # print("pred_class ?", pred_class)
        # print("pred_class type?", type(pred_class))

        # # 예측을 출력합니다.
        # return jsonify({"predict_class": classes[pred_class]})

if __name__ == '__main__':
    app.run()
