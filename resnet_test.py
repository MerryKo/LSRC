import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision
import cv2

print(torch.cuda.is_available())

input_path = "./dataset_dip/"
num_classes = 2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

model = models.resnet50(pretrained=False).to(device)
model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, num_classes)).to(device)
model.load_state_dict(torch.load('models/validation loss_ 0.0254  acc_ 0.9957best.h5'))

# Make predictions on sample test images
validation_img_paths = [
                        '1.jpg'
                        ]
# img_list = [Image.open(input_path + img_path) for img_path in validation_img_paths]
img_list = [Image.fromarray(cv2.imread(input_path + img_path)) for img_path in validation_img_paths]

validation_batch = torch.stack([data_transforms['validation'](img).to(device)
                                for img in img_list])

pred_logits_tensor = model(validation_batch)
print(pred_logits_tensor)

pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()
print(pred_probs)

fig, axs = plt.subplots(1, len(img_list), figsize=(20, 5))
for i, img in enumerate(img_list):
    ax = axs[i]
    ax.axis('off')
    ax.set_title("{:.0f}% NG, {:.0f}% OK".format(100*pred_probs[i,0],
                                                            100*pred_probs[i,1]))
    ax.imshow(img)
plt.show()