
import torch
import io
import torchvision
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

state_dict = torch.load("model_v1.pth", map_location=torch.device('cpu'))

class ImageClassificationBase(nn.Module):
  
  def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    my_transforms = transforms.Compose([transforms.Resize(128),
                                        #transforms.CenterCrop(),
                                        transforms.ToTensor(),
                                        ])
    return my_transforms(image).unsqueeze(0)
  

  def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
    
  def training_step(self,batch):
    images,labels = batch
    out = self(images)
    loss = F.cross_entropy(out,labels)
    return loss

  def validation_step(self,batch):
    images,labels = batch
    out = self(images)
    loss = F.cross_entropy(out,labels)
    acc = self.accuracy(out,labels)
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
    self.network.fc = nn.Linear(num_ftrs, 28)
    
  def forward(self,xb):
    out = self.network(xb)
    return out
  