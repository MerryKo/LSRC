import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision

# TODO https://www.kaggle.com/code/pmigdal/transfer-learning-with-resnet-50-in-pytorch

input_path = "./dataset_dip/"
num_classes = 2

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

data_transforms = {
    'train':
    transforms.Compose([
        transforms.Resize((45,640)),
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]),
    'validation':
    transforms.Compose([
        transforms.Resize((45,640)),
        transforms.ToTensor(),
        normalize
    ]),
}

image_datasets = {
    'train': 
    datasets.ImageFolder(input_path + 'train', data_transforms['train']),
    'validation': 
    datasets.ImageFolder(input_path + 'validation', data_transforms['validation'])
}

dataloaders = {
    'train':
    torch.utils.data.DataLoader(image_datasets['train'],
                                batch_size=10,
                                shuffle=True,
                                num_workers=0),  # for Kaggle
    'validation':
    torch.utils.data.DataLoader(image_datasets['validation'],
                                batch_size=10,
                                shuffle=False,
                                num_workers=0)  # for Kaggle
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = models.resnet50(pretrained=True).to(device)
    
for param in model.parameters():
    param.requires_grad = False   
    
model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, num_classes)).to(device)
# model.load_state_dict(torch.load('models/validation loss_ 0.5178  acc_ 0.7356best_0914.h5'))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr = 1e-6)


def train_model(model, criterion, optimizer, num_epochs=1000):
    checkpoint_fn = 'model_checkpoint.h5'
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            info = '{} loss_ {:.4f}  acc_ {:.4f}'.format(phase,
                                                        epoch_loss,
                                                        epoch_acc)
            print(info)
            # torch.save(model.state_dict(),'./models/'+info + checkpoint_fn)
            # try:
            #     # 存取數據
            #     if phase == 'train':
            #         train_losses.append(epoch_loss)
            #         train_accuracies.append(epoch_acc.item())
            #         # print(epoch_acc.item())
            #     else:
            #         val_losses.append(epoch_loss)
            #         val_accuracies.append(epoch_acc.item())
            #         # print(val_losses)

            #     # realtime 繪製
            #     plt.figure(figsize=(10,5))
            #     plt.subplot(1,2,1)
            #     plt.plot(train_losses, label = 'training loss', marker='o')
            #     plt.plot(val_losses, label = 'validation loss', marker='o')
            #     plt.xlabel('Epoch')
            #     plt.ylabel('Loss')
            #     plt.legend()
            #     plt.grid()

            #     plt.subplot(1,2,2)
            #     plt.plot(train_accuracies, label = 'training accuracy', marker='o')
            #     plt.plot(val_accuracies, label = 'validation accuracy', marker='o')
            #     plt.xlabel('Epoch')
            #     plt.ylabel('Accuracy')
            #     plt.legend()
            #     plt.grid()

            #     plt.tight_layout()
            #     plt.savefig('sum.png')
            #     # plt.show()
            # except:
            #     pass
            
            if phase =='validation' and epoch_loss < best_val_loss:
                print('Validation loss improved from {:.4f} to {:.4f}. Saving model...'.format(best_val_loss,epoch_loss))
                best_val_loss = epoch_loss
                torch.save(model.state_dict(),'./models/'+info+'best.h5')

    return model

model_trained = train_model(model, criterion, optimizer, num_epochs=3000)
torch.save(model_trained.state_dict(), './models/20230915_preCV_weights.h5')

