from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import random

class myimage:
    def __init__(self,img,method):
        self.method=method
        if method=='GRAY':
            #https://shengyu7697.github.io/python-opencv-rgb-to-gray/
            print("convert to gray")
            self.img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            self.img=img
            self.img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    def ReconizeFace(self):
        mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)
        boxes, _ = mtcnn.detect(img)
        cv2.rectangle(self.img,(int(boxes[0][0]),int(boxes[0][1])),(int(boxes[0][2]),int(boxes[0][3])),(0,0,255),10)
    def SaveImg(self):
        path = "./result2"
        if not os.path.isdir(path):
            os.makedirs(path)
        #https://www.delftstack.com/zh-tw/howto/matplotlib/matplotlib-display-image-in-grayscale/
        if myimg.method=='GRAY':
            plt.imshow(self.img,cmap='gray')
        else:
            plt.imshow(self.img)
        plt.axis('off')
        time_str=str(time.localtime( time.time() )[0])+str(time.localtime( time.time() )[1])+str(time.localtime( time.time() )[2])+str(time.localtime( time.time() )[3])+str(time.localtime( time.time() )[4])+str(time.localtime( time.time() )[5])
        random_n=str(int(100*random.random()))
        plt.savefig(path+"/result_"+args.method+"_"+time_str+"_"+random_n+".jpg")


parser = argparse.ArgumentParser(description='processing')
parser.add_argument('--input','--input',type=str,help='YOUR_IMAGE_PATH')
parser.add_argument('--method','--method',type=str,help='YOUR_METHOD')
args = parser.parse_args()

if os.path.isdir(args.input):
    print(os.listdir(args.input))
    for f in os.listdir(args.input):
        if f[-4:]=='.jpg':
            img = cv2.imread(args.input+"/"+f)
            myimg = myimage(img,args.method)
            myimg.ReconizeFace();
            myimg.SaveImg()
else:
    img = cv2.imread(args.input)
    myimg = myimage(img,args.method)
    myimg.ReconizeFace();
    myimg.SaveImg()
