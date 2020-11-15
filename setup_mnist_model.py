import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from torchvision import transforms,datasets

class MNIST(nn.Module):
  def __init__(self):
    super(MNIST, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, 1)
    self.conv2 = nn.Conv2d(32, 32, 3, 1)
    self.conv3= nn.Conv2d(32, 64, 3, 1)
    self.conv4= nn.Conv2d(64, 64, 3, 1)
    self.dropout1 = nn.Dropout2d(0.5)
    self.fc1 = nn.Linear(1024, 200)
    self.fc2 = nn.Linear(200, 200)
    self.fc3 = nn.Linear(200, 10)

  def forward(self, x):
    x = self.conv1(x)   
    x = F.relu(x)
    x = self.conv2(x)   
    x = F.relu(x)
    x = F.max_pool2d(x, 2)  
    x = self.conv3(x)       
    x = F.relu(x)
    x = self.conv4(x)
    x = F.relu(x)
    x = F.max_pool2d(x, 2)  
    x = torch.flatten(x,1)  
    x = self.fc1(x)         
    x = F.relu(x)
    x = self.dropout1(x)
    x = self.fc2(x)
    x = F.relu(x)
    x = self.fc3(x)        
    return x

def fit(model,device,train_loader,val_loader,optimizer,criterion,epochs):
  data_loader = {'train':train_loader,'val':val_loader}
  print("Fitting the model...")
  train_loss,val_loss=[],[]
  train_acc,val_acc=[],[]
  for epoch in range(epochs):
    loss_per_epoch,val_loss_per_epoch=0,0
    acc_per_epoch,val_acc_per_epoch,total,val_total=0,0,0,0
    for phase in ('train','val'):
      for i,data in enumerate(data_loader[phase]):
        inputs,labels  = data[0].to(device),data[1].to(device)
        outputs = model(inputs)
        #preding classes of one batch
        preds = torch.max(outputs,1)[1]
        #calculating loss on the output of one batch
        loss = criterion(outputs,labels)
        if phase == 'train':
          acc_per_epoch+=(labels==preds).sum().item()
          total+= labels.size(0)
          optimizer.zero_grad()
          #grad calc w.r.t Loss func
          loss.backward()
          #update weights
          optimizer.step()
          loss_per_epoch+=loss.item()
        else:
          val_acc_per_epoch+=(labels==preds).sum().item()
          val_total+=labels.size(0)
          val_loss_per_epoch+=loss.item()
    print("Epoch: {} Loss: {:0.6f} Acc: {:0.6f} Val_Loss: {:0.6f} Val_Acc: {:0.6f}".format(epoch+1,loss_per_epoch/len(train_loader),acc_per_epoch/total,val_loss_per_epoch/len(val_loader),val_acc_per_epoch/val_total))
    train_loss.append(loss_per_epoch/len(train_loader))
    val_loss.append(val_loss_per_epoch/len(val_loader))
    train_acc.append(acc_per_epoch/total)
    val_acc.append(val_acc_per_epoch/val_total)
  return train_loss,val_loss,train_acc,val_acc

if __name__=='__main__':
  np.random.seed(42)
  torch.manual_seed(42)

  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
  dataset = datasets.MNIST(root = './data', train=True, transform = transform, download=True)
  train_set, val_set = torch.utils.data.random_split(dataset, [55000, 5000])
  train_loader = torch.utils.data.DataLoader(train_set,batch_size=128,shuffle=True)
  val_loader = torch.utils.data.DataLoader(val_set,batch_size=128,shuffle=True)
  
  use_cuda=True
  device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

  model = MNIST().to(device)
  summary(model,(1,28,28))

  optimizer = optim.SGD(model.parameters(),lr=0.01, momentum=0.9, nesterov=True, weight_decay=1e-6)
  criterion = nn.CrossEntropyLoss()

  train_loss,val_loss,train_acc,val_acc=fit(model,device,train_loader,val_loader,optimizer,criterion,50)

  fig = plt.figure(figsize=(5,5))
  plt.plot(np.arange(1,51), train_loss, "*-",label="Training Loss")
  plt.plot(np.arange(1,51), val_loss,"o-",label="Val Loss")
  plt.xlabel("Num of epochs")
  plt.legend()
  plt.savefig('mnist_model_loss_event.png')

  fig = plt.figure(figsize=(5,5))
  plt.plot(np.arange(1,51), train_acc, "*-",label="Training Acc")
  plt.plot(np.arange(1,51), val_acc,"o-",label="Val Acc")
  plt.xlabel("Num of epochs")
  plt.legend()
  plt.savefig('mnist_model_accuracy_event.png')

  torch.save(model.state_dict(),'./models/mnist_model.pt')
