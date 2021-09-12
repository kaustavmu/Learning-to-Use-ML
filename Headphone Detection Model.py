# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 15:52:31 2021

@author: Kaustav Mukherjee
"""
#This program is intended to create a machine learning model that can determine whether its creator, Kaustav Mukherjee is wearing a headphone in a picture.

import cv2
import os
import time
import sys

from PIL import Image
import PIL
import glob

import numpy
import matplotlib.pyplot as plt
import math

from torchvision import transforms
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

import mediapipe as mp

mpfd = mp.solutions.face_detection
mpd = mp.solutions.drawing_utils

torch.set_printoptions(threshold=10_000)

try:
    while True:
        print("Please select what you would like to do.")
        print("1 - Capture Headphone Images")
        print("2 - Capture Headphone-less Images")
        print("3 - Train the Model")
        print("4 - Use Previous Model")
        
        action = input("Enter the appropriate number:")
        
        if action == "1":
            
            path = "C:/Users/Kaustav Mukherjee/Documents/Headphone Model Data Collection/Headphones On"
            cap=cv2.VideoCapture(0)
            i = len([i for i in os.listdir(path)])
            #i = 0
            while (cap.isOpened()):
                ret, frame = cap.read()
                if ret == False:
                    break
                
                with mpfd.FaceDetection(model_selection = 0, min_detection_confidence = 0.5) as fd:
            
                    frame.flags.writeable = False
                    results = fd.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    
                    frame.flags.writeable = True        
                    if results.detections:
                        for detection in results.detections:
                            height, width, channels = frame.shape
                            rbb = detection.location_data.relative_bounding_box
                            X = math.floor((rbb.xmin*0.9)*width)
                            Y = math.floor((rbb.ymin*0.9)*height)
                            W = math.floor((rbb.width + 0.2*rbb.xmin)*width)
                            H = math.floor((rbb.height + 0.2*rbb.ymin)*height)
                            frame = frame[Y:Y+H, X:X+W] 
                            
                cv2.imwrite(os.path.join(path, 'Frame'+str(i)+'.jpg'), frame)
                i += 1
            cap.release()
            cv2.destroyAllWindows()
            
        elif action == "2":
            
            path = "C:/Users/Kaustav Mukherjee/Documents/Headphone Model Data Collection/Headphones Off"
            cap=cv2.VideoCapture(0)
            i = len([i for i in os.listdir(path)])
            #i=0
            while (cap.isOpened()):
                ret, frame = cap.read()
                if ret == False:
                    break
                
                with mpfd.FaceDetection(model_selection = 0, min_detection_confidence = 0.5) as fd:
            
                    frame.flags.writeable = False
                    results = fd.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    
                    frame.flags.writeable = True        
                    if results.detections:
                        for detection in results.detections:
                            height, width, channels = frame.shape
                            rbb = detection.location_data.relative_bounding_box
                            X = math.floor((rbb.xmin*0.9)*width)
                            Y = math.floor((rbb.ymin*0.9)*height)
                            W = math.floor((rbb.width + 0.2*rbb.xmin)*width)
                            H = math.floor((rbb.height + 0.2*rbb.ymin)*height)
                            frame = frame[Y:Y+H, X:X+W] 
                            
                cv2.imwrite(os.path.join(path, 'Frame'+str(i)+'.jpg'), frame)
                i += 1
            cap.release()
            cv2.destroyAllWindows()
            
        elif action == "3":
            
            # Compiling and processing the images from the folder to create datasets and dataloaders.
            
            tensortransform = transforms.ToTensor()
            directory1 = "C:/Users/Kaustav Mukherjee/Documents/Headphone Model Data Collection/Headphones On"
            directory2 = "C:/Users/Kaustav Mukherjee/Documents/Headphone Model Data Collection/Headphones Off"
            paths = []
            x_data = []
            i = 0
            j = 0
            
            for filename in os.scandir(directory1):
                paths.append(filename.path)
                i += 1
            for filename in os.scandir(directory2):
                paths.append(filename.path)
                j += 1
                
            y_tensor = torch.cat((torch.ones(i, dtype=torch.float), torch.zeros(j, dtype=torch.float)))
    
            for path in paths:
                image = Image.open(path)
                x_data.append(tensortransform(image.resize((40, 30))))
            x_tensor = torch.stack((x_data))
    
            dataset = TensorDataset(x_tensor, y_tensor)
            dssize = i + j
            print(dssize)
            traindssize = int(dssize*0.8)
            valdssize = dssize - traindssize
            print(traindssize, valdssize)
            TrainDS, ValDS = random_split(dataset, [traindssize, valdssize])
    
            batch_size = 16
            TrainDL = DataLoader(TrainDS, batch_size, shuffle = True)
            ValDL = DataLoader(ValDS, batch_size)
            
            def get_default_device():
                """Pick GPU if available, else CPU"""
                if torch.cuda.is_available():
                    return torch.device('cuda')
                else:
                    return torch.device('cpu')
                
            device = get_default_device()
            print(device)
                
            def to_device(data, device):
                """Move tensor(s) to chosen device"""
                if isinstance(data, (list,tuple)):
                    return [to_device(x, device) for x in data]
                return data.to(device, non_blocking=True)
            
            class DeviceDataLoader():
                """Wrap a dataloader to move data to a device"""
                def __init__(self, dl, device):
                    self.dl = dl
                    self.device = device
                    
                def __iter__(self):
                    """Yield a batch of data after moving it to device"""
                    for b in self.dl: 
                        yield to_device(b, self.device)
            
                def __len__(self):
                    """Number of batches"""
                    return len(self.dl)
                
            TrainDL = DeviceDataLoader(TrainDL, device)
            ValDL = DeviceDataLoader(ValDL, device)
            
            # Defining the model.
            
            def closer(a):
                x = [abs(a), abs(a-1)]
                return x.index(min(x))
            
            def accuracy(pred, yb):
                approximations = [closer(i[0]) for i in pred] 
                l = len(approximations)
                return torch.tensor(sum([approximations[i]==yb[i] for i in range(l)]) / l)
            
            def evaluate(model, val_loader):
                outputs = [model.evaluationstep(batch) for batch in val_loader]
                return model.validation_epoch_end(outputs)
            
            class MnistModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    #self.linear1 = nn.Linear(ins, hs1)
                    #self.linear2 = nn.Linear(hs1, os)
                    self.network = nn.Sequential(
                        nn.Conv2d(3, 32, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2), # output: 64 x 20 x 15
            
                        nn.Flatten(), 
                        nn.Linear(64*20*15, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1))
                    
                def forward(self, xb):
                    #xb = xb.reshape(xb.size(0), -1)
                    #out = self.linear1(xb)
                    #out = F.relu(out)
                    #out = self.linear2(out)
                    #return out
                    return self.network(xb)
                
                def training_step(self, batch):
                    xb, yb = batch
                    out = self(xb)
                    yb = yb.view(-1, 1)
                    loss = F.binary_cross_entropy_with_logits(out, yb)
                    return loss
                
                def evaluationstep(self, batch):
                    xb, yb = batch
                    out = self(xb)
                    yb = yb.view(-1, 1)
                    loss = F.binary_cross_entropy_with_logits(out, yb)
                    acc = accuracy(out, yb)
                    return {'val_loss': loss, 'val_acc': acc}
                    
                def validation_epoch_end(self, outputs):
                    batch_losses = [x['val_loss'] for x in outputs]
                    epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
                    batch_accs = [x['val_acc'] for x in outputs]
                    epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
                    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
                
                def epoch_end(self, epoch, result):
                    print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
                
            model = MnistModel()
            to_device(model, device)
            
            epochs = 10
            
            lr = 0.1
            
            def fit(epochs, lr, model, TrainDL, ValDL, opt_func=torch.optim.SGD):
                optimizer = opt_func(model.parameters(), lr)
                history = [] # for recording epoch-wise results
                
                for epoch in range(epochs):
                    
                    # Training Phase 
                    for batch in TrainDL:
                        loss = model.training_step(batch)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    # Validation phase
                    result = evaluate(model, ValDL)
                    model.epoch_end(epoch, result)
                    history.append(result)
            
                return history
                            
            fit(epochs, lr, model, TrainDL, ValDL)
            path = "C:/Users/Kaustav Mukherjee/Documents/Headphone Model Data Collection/Models"
            modelnumber = str(len([i for i in os.listdir(path)]))
            torch.save(model.state_dict(), path+"/"+modelnumber)
            
            to_device(model, 'cpu')
            
            cap=cv2.VideoCapture(0)
            
            def predict_image(img, model):
                    xb = img.unsqueeze(0)
                    yb = model(xb)
                    return closer(yb)
            
            while (cap.isOpened()):
                ret, frame = cap.read()
                if ret == False:
                    break
                
                with mpfd.FaceDetection(model_selection = 0, min_detection_confidence = 0.5) as fd:
            
                    frame.flags.writeable = False
                    results = fd.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    
                    frame.flags.writeable = True        
                    if results.detections:
                        for detection in results.detections:
                            height, width, channels = frame.shape
                            rbb = detection.location_data.relative_bounding_box
                            X = math.floor((rbb.xmin*0.9)*width)
                            Y = math.floor((rbb.ymin*0.9)*height)
                            W = math.floor((rbb.width + 0.2*rbb.xmin)*width)
                            H = math.floor((rbb.height + 0.2*rbb.ymin)*height)
                            frame = frame[Y:Y+H, X:X+W]
                    else:
                        X, Y, W, H = 0, 0, 0, 0
                            
                if (X*Y*H*W) > 0:
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    im_pil = Image.fromarray(img)
                    testimg = tensortransform(im_pil.resize((40, 30)))
                    print("Headphones On" if predict_image(testimg, model) else "Headphones Off") 
                else:
                    print("Image Unclear")
                time.sleep(1)
                
            cap.release()
            cv2.destroyAllWindows()
            
        elif action == "4":
            
            tensortransform = transforms.ToTensor()
            
            class MnistModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    #self.linear1 = nn.Linear(ins, hs1)
                    #self.linear2 = nn.Linear(hs1, os)
                    self.network = nn.Sequential(
                        nn.Conv2d(3, 32, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2), # output: 64 x 20 x 15
            
                        nn.Flatten(), 
                        nn.Linear(64*20*15, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1))
                    
                def forward(self, xb):
                    #xb = xb.reshape(xb.size(0), -1)
                    #out = self.linear1(xb)
                    #out = F.relu(out)
                    #out = self.linear2(out)
                    #return out
                    return self.network(xb)
            
            model = MnistModel()
            
            path = "C:/Users/Kaustav Mukherjee/Documents/Headphone Model Data Collection/Models"
            modelnumber = str(len([i for i in os.listdir(path)])-1)
            model.load_state_dict(torch.load(path+"/"+modelnumber))
            
            cap=cv2.VideoCapture(0)
            
            def closer(a):
                x = [abs(a), abs(a-1)]
                return x.index(min(x))
            
            def predict_image(img, model):
                    xb = img.unsqueeze(0)
                    yb = model(xb)
                    return closer(yb)
            
            while (cap.isOpened()):
                ret, frame = cap.read()
                if ret == False:
                    break
                
                with mpfd.FaceDetection(model_selection = 0, min_detection_confidence = 0.5) as fd:
            
                    frame.flags.writeable = False
                    results = fd.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    
                    frame.flags.writeable = True        
                    if results.detections:
                        for detection in results.detections:
                            height, width, channels = frame.shape
                            rbb = detection.location_data.relative_bounding_box
                            X = math.floor((rbb.xmin*0.9)*width)
                            Y = math.floor((rbb.ymin*0.9)*height)
                            W = math.floor((rbb.width + 0.2*rbb.xmin)*width)
                            H = math.floor((rbb.height + 0.2*rbb.ymin)*height)
                            cutframe = frame[Y:Y+H, X:X+W]
                    else:
                        X, Y, W, H = 0, 0, 0, 0
                            
                cv2.imshow('MediaPipe Face Detection', frame)
                if cv2.waitKey(5) & 0xFF == 27:
                    break        
                
                if (X*Y*H*W) > 0:
                    img = cv2.cvtColor(cutframe, cv2.COLOR_BGR2RGB)
                    im_pil = Image.fromarray(img)
                    testimg = tensortransform(im_pil.resize((40, 30)))
                    print("Headphones On" if predict_image(testimg, model) else "Headphones Off") 
                else:
                    print("Image Unclear")
                
            cap.release()
            cv2.destroyAllWindows()
            
        break
    
except KeyboardInterrupt:
    cap.release()
    sys.exit()
   
