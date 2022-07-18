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

img = cv2.imread('./input/1.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20) # initializing mtcnn for face detection
resnet = InceptionResnetV1(pretrained='vggface2').eval() # initializing resnet for face img to embeding conversion

boxes, _ = mtcnn.detect(img)
cv2.rectangle(img,(int(boxes[0][0]),int(boxes[0][1])),(int(boxes[0][2]),int(boxes[0][3])),(255,0,0),10)
plt.imshow(img)
plt.axis('off')
plt.savefig('result_1.jpg')