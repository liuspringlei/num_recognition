"""
使用笔记本电脑摄像头测试训练模型
"""

import numpy as np 
import torch 
import cv2 as cv 
import torch.nn as nn 
from model import convnet


def Contrast_and_Brightness(alpha, beta, img):  #增强对比度，让图像在识别的过程中更加的清楚
    blank = np.zeros(img.shape, img.dtype)
    # dst = alpha * img + (1-alpha) * blank + beta
    dst = cv.addWeighted(img, alpha, blank, 1-alpha, beta)
    return dst

cap = cv.VideoCapture(0) #打开摄像头
while (cap.isOpened()): #如果摄像头启动，则进行while循环
    ret ,frame = cap.read()
    frame=Contrast_and_Brightness(1, 1, frame) #第一个参数调整对比度，第二个参数调整亮度，有时候对比度或者亮度太暗影响识别
    img_gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    imgGS = cv.GaussianBlur(img_gray,(5,5),0)
    erosion = cv.erode(imgGS,(3,3),iterations = 3)
    dilate = cv.dilate(erosion,(3,3),iterations=3)
    edge = cv.Canny(dilate,80,200,255)
    contours,hierachy= cv.findContours(edge,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

    digitcnts= []
    for i in contours:
        (x,y,w,h) = cv.boundingRect(i)
        if w <100 and h >45 and h <160:
            digitcnts.append(i)
    m = 0
    for c in digitcnts:
        (x,y,w,h) = cv.boundingRect(c)
        m += 1 
        roil = frame [y:y+h , x:x+w]
        height ,width,channel = roil.shape
        for i in range(height):
            for j in range (width):
                b,g,r = roil [i, j]
                if g >180:
                    b=255
                    r=255
                    g=255 
                else :
                    b=0
                    g=0
                    r=0
                roil[i,j]=[b,g,r]
        roil =255-roil 
        roil2 = cv.copyMakeBorder(roil,30,30,30,30,cv.BORDER_CONSTANT,value=[0,0,0])
        cv.imwrite(".\\learn\\num_recognition_cnn\\rec_cache\\%d.png"%m,roil2) #边缘检测后的提取的照片路径
        img1 =cv.imread(".\\learn\\num_recognition_cnn\\rec_cache\\%d.png"%m,0) #识别后的照片缓存的保存路径
        img1 = cv.GaussianBlur(img1,(5,5),0)
        img1 = cv.dilate(img1,(3,3),iterations=3)
        img2 = cv.resize(img1,(28,28),interpolation=cv.INTER_CUBIC)
        img3 = np.array(img2)/255
        img4 = np.reshape(img3,[-1,784])

        images =torch.tensor(img4,dtype=torch.float32)
        images = images.resize(1,1,28,28)

        model =convnet()
        model.load_state_dict(torch.load('convnet.cpkl')) #加载训练好的模型文件
        model.eval()
        outputs = model(images)
        values,indices = outputs.data.max(1)

        font = cv.FONT_HERSHEY_SIMPLEX
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),0)
        cv.putText(frame,str(indices[0].item()),(x,y),font,1,(0,0,255),1,cv.LINE_AA)
        cv.imshow("capture",frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()