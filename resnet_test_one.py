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
import glob

def dip(f):
    image = cv2.imread(f)
    edges = cv2.Canny(image, 50, 95)
    
    # 使用Hough Line Transform檢測直線
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    # 找到最下面的直線
    if lines is not None:
        # 將直線根據Y座標進行排序
        lines = sorted(lines, key=lambda line: line[0][1], reverse=True)
        
        # 繪製最下面的直線（假設只有一條）
        x1, y1, x2, y2 = lines[0][0]
        print(x1, y1, x2, y2)
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow('crop1',image)
        cv2.waitKey(0)

    img = image[y1-60:y1-15,:]
    print(img.shape)
    alpha = 100.5 #對比
    beta = 12  #亮度
    img = cv2.GaussianBlur(img,(5,5),1)
    cv2.imshow('crop1',img)
    cv2.waitKey(0)
    enhance = cv2.convertScaleAbs(img, alpha, beta)
    cv2.imshow('enhance',enhance)
    cv2.waitKey(0)
    return enhance

def load_model():
    return

def test(img_f):
    num_classes = 2

    # 创建ResNet-50模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=False).to(device)
    model.fc = nn.Sequential(
                nn.Linear(2048, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, num_classes)).to(device)
    model.load_state_dict(torch.load('models/validation loss_ 0.0254  acc_ 0.9957best.h5'))
    model.eval()  # 将模型设置为评估模式
    transform = transforms.Compose([
        transforms.Resize((45,640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mypath = glob.glob("./ng/*jpg")
    for f in mypath:
        # 加载待预测的图像
        # image = Image.open(f)
        image = Image.fromarray(dip(f))
        image = transform(image).to(device)
        image = image.unsqueeze(0)  # 添加一个维度，以匹配模型的输入形状

        pred_logits_tensor = model(image)
        # print(pred_logits_tensor)
        pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()
        # print(pred_probs)
        print("{:.0f}% NG, {:.0f}% OK".format(100*pred_probs[0,0],
                                                            100*pred_probs[0,1]))

test('')