import torch
from torchvision import models
import torch.nn as nn

def resnet50_evalgenerator(device, path):
    
    resnet50_model = models.resnet50()
    resnet50_model.conv1 = nn.Conv2d(15, 64, kernel_size=(7, 7), stride=(2, 2), 
                                     padding=(3, 3), bias=False)
    num_classes = 48
    resnet50_model.fc = nn.Linear(resnet50_model.fc.in_features, num_classes)

    # Move the model to the device
    resnet50_model = resnet50_model.to(device)

    # import the model weights from trained model resnet50_classifer.py
    resnet50_model.load_state_dict(torch.load(path + 'resnet50_48classes.pth'))
    
    return resnet50_model