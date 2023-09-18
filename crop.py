'''
import cv2
import glob


mypath = glob.glob('./dataset_cvtest/train/ok/*.jpg')
print(mypath)
for img_fn in mypath:
    # img_fn = './dataset_crop/train/ng/001840158_CFPKL504_E8_1_20230829001840511_P136.3_OK_T32_Q1.jpg'
    img = cv2.imread(img_fn)
    # crop = img[200:310, :640]
    # cv2.imwrite(img_fn,crop)
    alpha = 10.5 #對比
    beta = 7  #亮度
    enhance_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    cv2.imwrite(img_fn, enhance_img)
    # cv2.imshow('crop',crop)
    # cv2.waitKey(0)
    # print(img.shape)
    # print(crop.shape)
'''

# alpha = 10.5 #對比
# beta = 7  #亮度

# image = cv2.imread('./dataset_crop/train/ok/001011476_CFPKL504_E8_1_20230906001011632_P38.3_OK_T9_Q7.jpg')
# enhance_img = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# image = cv2.imread('./dataset_crop/train/ng/001840158_CFPKL504_E8_1_20230829001840256_P38.3_OK_T9_Q1.jpg')
# enhance_img1 = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
# cv2.imshow('crop',enhance_img)
# cv2.imshow('crop1',enhance_img1)
# cv2.waitKey(0)


import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

def crop_vip(f):

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

    img = image[y1-60:y1-15,:]
    print(img.shape)
    alpha = 100.5 #對比
    beta = 12  #亮度
    img = cv2.GaussianBlur(img,(5,5),1)
    en = cv2.convertScaleAbs(img, alpha, beta)
    cv2.imwrite(f, en)


def data_aug(f):
    img = cv2.imread(f)
    flip0 = cv2.flip(img, 0)
    flip1 = cv2.flip(img, 1)
    flip_1 = cv2.flip(img, -1)
    cv2.imwrite(f+'flip0.jpg',flip0)
    cv2.imwrite(f+'flip1.jpg',flip1)
    cv2.imwrite(f+'flip_1.jpg',flip_1)


mypath = glob.glob('./dataset_dip/train/ng/*.jpg')
for f in mypath:
    # crop_vip(f) #1
    data_aug(f) #2


    
