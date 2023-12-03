import torch
import torchvision
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F



def accuracy(outputs, labels):
  _, preds = torch.max(outputs, dim=1)
  return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# 모델 로드
state_dict1 = torch.load("./models/s1_binary_class.pth", map_location=torch.device('cpu'))
state_dict2 = torch.load("./models/s3_disease_class.pth", map_location=torch.device('cpu'))

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

# binary classification
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
  
# disease classification  
class Plant_Disease_Model2(ImageClassificationBase):
  
  def __init__(self):
    super().__init__()
    self.network = models.resnet34(weights=None)

    num_ftrs = self.network.fc.in_features
    self.network.fc = nn.Linear(num_ftrs, 28)
    
  def forward(self,xb):
    out = self.network(xb)
    return out
  
model1 = Plant_Disease_Model()
model2 = Plant_Disease_Model2()

model1.load_state_dict(state_dict1)
model1.eval()

model2.load_state_dict(state_dict2)
model2.eval()

classes = ['bean__bean_spot', 'bean__blight', 'bean__brown_spot', 'bean__healthy', 'corn__common_rust', 'corn__gray_spot', 'corn__healthy', 'green_onion__black_spot', 'green_onion__downy_mildew', 'green_onion__healthy', 'green_onion__rust', 'lectuce__downy_mildew', 'lectuce__drop', 'lectuce__healthy', 'pepper__anthracnose', 'pepper__healthy', 'pepper__powdery_mildew', 'potato__Early_Blight', 'potato__healthy', 'potato__late_Blight', 'potato__soft_rot', 'pumpkin__healthy', 'pumpkin__leaf_mold', 'pumpkin__mosaic', 'pumpkin__powdery_mildew', 'radish__black_spot', 'radish__downy_mildew', 'radish__healthy']

# 이미지를 전처리
transform = transforms.Compose([
    transforms.Resize(size=128),
    transforms.ToTensor(),
])

with torch.no_grad():
  image = Image.open("test2.jpg")
  image = transform(image)
  #image = image[:,:128,:]
  image = torch.unsqueeze(image, 0)
  output1 = model1(image)
  pred_class = output1.argmax()
  print(pred_class)
  
  if pred_class.item() == 0: # 식물 아닌 경우
    print("식물 아님")
    
  if pred_class.item() == 1: # 식물인경우
    print("식물임")

    output2 = model2(image)
    _, preds = torch.topk(output2, 2)
    probs = torch.nn.functional.softmax(output2, dim=1)[0,  preds[0]]
    print(preds, probs)
    print(probs.shape)

    top_classes = preds.tolist()
    print(top_classes)

    for i in range(2):
      print(classes[top_classes[0][i]])

