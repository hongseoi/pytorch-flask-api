import torch
import torchvision
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F


# 이미지를 불러오기
image = Image.open("test3.jpg")

# 이미지를 전처리
transform = transforms.Compose([
    transforms.Resize(size=128),
    transforms.ToTensor(),
])

image = transform(image)
image = image[:,:128,:]
image = torch.unsqueeze(image, 0)



def accuracy(outputs, labels):
  _, preds = torch.max(outputs, dim=1)
  return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# 모델 로드
state_dict = torch.load("plantDisease-resnet34-v1.pth", map_location=torch.device('cpu'))

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
    self.network.fc = nn.Linear(num_ftrs, 38)
    
  def forward(self,xb):
    out = self.network(xb)
    return out
  
model = Plant_Disease_Model2()
model.load_state_dict(state_dict)
model. eval()

classes = ['bean__bean_spot', 'bean__blight', 'bean__brown_spot', 'bean__healthy', 'corn__common_rust', 'corn__gray_spot', 'corn__healthy', 'green_onion__black_spot', 'green_onion__downy_mildew', 'green_onion__healthy', 'green_onion__rust', 'lectuce__downy_mildew', 'lectuce__drop', 'lectuce__healthy', 'pepper__anthracnose', 'pepper__healthy', 'pepper__powdery_mildew', 'potato__Early_Blight', 'potato__healthy', 'potato__late_Blight', 'potato__soft_rot', 'pumpkin__healthy', 'pumpkin__leaf_mold', 'pumpkin__mosaic', 'pumpkin__powdery_mildew', 'radish__black_spot', 'radish__downy_mildew', 'radish__healthy']

# 추론을 수행
output = model(image)

# 가장 높은 확률을 가진 클래스를 예측합니다.
pred_class = output.argmax()

# 예측을 출력합니다.
print("예측 클래스:", classes[pred_class])
