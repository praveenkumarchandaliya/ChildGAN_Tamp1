# -*- coding:utf-8 -*-
# Author: Praveen Kumar Chandaliya 
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pickle
import numpy as np
from torch import autograd
from misc import *
from PIL import ImageFile


import os.path
import torch.nn.parallel
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
from PIL import Image
import PIL
import random
from torchvision.utils import save_image
import cv2
import shutil
torch.cuda.set_device('cuda:0')
layer_names = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
               'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
               'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
               'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
               'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5']
               
content_layers = ['relu1_1', 'relu2_1', 'relu3_1']

use_cuda = torch.cuda.is_available()
n_z = 50
n_l = 5
n_channel = 3
n_disc = 16
n_gen = 64
nef = 64
ndf = 64
ngpu = 1
n_z = 50
n_l = 5
n_channel = 3
n_disc = 16
n_gen = 64
n_age = int(n_z/n_l) #12
n_gender = int(n_z/2) #25
image_size = 128
nz = int(n_z)
nef = int(nef)
ndf = int(ndf)
nc = 3
out_size = image_size // 16  # 64
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Self Attection Block
class Self_Attn(nn.Module):
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.action = activation
        self.query_conv = nn.Conv2d(in_channels=in_dim,out_channels=in_dim//8,kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim,out_channels=in_dim//8,kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim,out_channels=in_dim,kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=1)
    def forward(self,x):
        m_batchsize,C,width,height = x.size()
        
        proj_query = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1)
        proj_key = self.key_conv(x).view(m_batchsize,-1,width*height)
        energy = torch.bmm(proj_query,proj_key) #batch matrix-matrix product of matrices store
        attention = self.softmax(energy)
        proj_value  = self.value_conv(x).view(m_batchsize,-1,width*height)
        out  = torch.bmm(proj_value,attention.permute(0,2,1))
        out  = out.view(m_batchsize,C,width,height)
        out = self.gamma*out +x
        return out,attention


# Residule Block
class resnet_block(nn.Module):
    def __init__(self, channel, kernel, stride, padding):
        super(resnet_block, self).__init__()
        self.channel = channel
        self.kernel = kernel
        self.strdie = stride
        self.padding = padding
        self.conv1 = nn.Conv2d(channel, channel, kernel, stride, padding)
        self.conv1_norm = nn.BatchNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel, stride, padding)
        self.conv2_norm = nn.BatchNorm2d(channel)
        #self.initialize_weights()

    def forward(self, input):
        x = F.relu(self.conv1_norm(self.conv1(input)), True)
        x = self.conv2_norm(self.conv2(x))

        return input + x  # Elementwise Sum


# Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, nef, 4, 2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(nef, nef * 2, 4, 2, padding=1),
            nn.ReLU(True),
        )
        self.encoder_second = nn.Sequential(
            nn.Conv2d(nef * 2, nef * 4, 4, 2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(nef * 4, nef * 8, 4, 2, padding=1),
            nn.ReLU(True),
        )
        self.resnet_blocks = []
        for i in range(9):
            self.resnet_blocks.append(resnet_block(nef * 2, 3, 1, 1))

        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)
        self.attn1  = Self_Attn(512,'relu')
        self.mean = nn.Linear(nef * 8 * out_size * out_size, nz)
        self.logvar = nn.Linear(nef * 8 * out_size * out_size, nz)

    def sampler(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        if use_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

    def forward(self, input):
        batch_size = input.size(0)
        
        hidden = self.encoder(input)
        hidden = self.resnet_blocks(hidden)
        hidden = self.encoder_second(hidden)
        
        out,ep1  = self.attn1(hidden)
        
        hidden = out.view(batch_size, -1)
        mean, logvar = self.mean(hidden), self.logvar(hidden)
        latent_z = self.sampler(mean, logvar)
        return latent_z,ep1
encoder = Encoder()

# Decoder 
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
 
        self.decoder_dense = nn.Sequential(
            nn.Linear(n_z+n_l*n_age+n_gender, ndf * 8 * out_size * out_size),
            nn.ReLU(True)
        )
        self.decoder_conv = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf * 8, ndf * 4, 3, padding=1),
            nn.ReLU(True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf * 4, ndf * 2, 3, padding=1),
            nn.ReLU(True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf * 2, ndf, 3, padding=1),
            nn.ReLU(True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf, nc, 3, padding=1),
            nn.Tanh()

        )

    def forward(self, z,age,gender):
        batch_size = z.size(0)
        l = age.repeat(1, n_age)  # size = 20 * 48
        k = gender.view(-1, 1).repeat(1, n_gender)  # size = 20 * 25
        x = torch.cat([z, l, k.float()], dim=1)  # size = 20 * 123
        hidden = self.decoder_dense(x).view(batch_size,ndf * 8, out_size, out_size)
        output = self.decoder_conv(hidden)
        return output

# Check the cuda
if use_cuda:
    encoder = Encoder().cuda()
    decoder = Decoder().cuda()
    
    
# Load the train encoder and decoder model weights    
c = torch.load('encoder_epoch_21000.pth')
encoder.load_state_dict(c)
d = torch.load('decoder_epoch_21000.pth')
decoder.load_state_dict(d)

outf="Result/"
if not os.path.exists(outf):
    os.mkdir(outf)

def get_loader(img_dir,label,batch_size=2,img_size=128,mode="train",num_workers=1):
    transform=[]
    transform.append(transforms.Resize(img_size))
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = transforms.Compose(transform)
    file_list=[]
    for dir in os.listdir(img_dir):
        print(dir)
        path, dirs, files = next(os.walk((os.path.join(img_dir, dir))))
        print("files",files)
        if (len(files) == 1):
            pass
        else:
            for file in os.listdir(os.path.join(img_dir, dir)):
                file_list.append(os.path.join(img_dir, dir)+"/"+file)

    filenames = []
    unchanged_target_ages = []
    images = []
    targets = []
    target_genders = np.ones((len(file_list), 1), dtype=np.int32) * -1
    for line in file_list:

        image = Image.open(line)
        image = transform(image)
        images.append(np.array(image))
    images = np.array(images)
    return  images,file_list

#  Test is directory
images,file_list=  get_loader(img_dir="Test",img_size=128,label=0,batch_size=8)
print(file_list)
images = torch.FloatTensor(images)
batch_size = 8
import math
import scipy.misc
num_batches = int(math.ceil(images.shape[0]/batch_size))
i=0
for batch in range(num_batches):
    batch_image = images[batch*batch_size:(batch+1)*batch_size,:,:,:].repeat(5, 1, 1, 1).cuda()
    file_batch = file_list[batch*batch_size:(batch+1)*batch_size]
    fixed_l = -torch.ones(40 * 5).view(40, 5)
    for i, l in enumerate(fixed_l):
        l[i // 8] = 1
    fixed_z,ep1 = encoder(batch_image)
    target_genders = -torch.ones(batch_size*1).view(batch_size,1)
    fcount=0
    for file_name  in file_batch:
        print(file_name)
        
        target_gender = int(file_name.split("/")[-1].split("_")[2])
        print(file_name,":",target_gender)
        
        if (target_gender == 1):
            target_genders[fcount] = 1
        fcount=fcount+1
    fixed_g= target_genders.view(-1, 1).repeat(5, 1)
    fixed_g_v = Variable(fixed_g).cuda()
    fixed_l_v = Variable(fixed_l).cuda()
    fixed_fake = decoder(fixed_z,fixed_l_v,fixed_g_v)
    outputpath = outf+"/"+str(batch)+".jpg"
    vutils.save_image(fixed_fake.data,outputpath , normalize=True)
    print("output path",outputpath)
    img = Image.open(outputpath)
    noOfRow = 5
    noOfColumn = 8
    x1 = 2
    y1 = 2
    x2 = 130
    y2 = 130
    folder = file_batch
    # Store result according the test image id.
    for i in range(0, noOfColumn):
        dest_dir = file_batch[i].split("/")[-1]
        if not os.path.exists(outf+"/"+dest_dir):
            os.mkdir(outf+"/"+dest_dir)
        for j in range(1, noOfRow + 1):
            area = (x1, y1, x2, y2)
            cropped_img = img.crop(area)
            imgName = "{}{}".format(i, j)
            if(int(imgName)==1 or int(imgName)==11 or int(imgName)==21 or int(imgName)==31 or int(imgName)==41 or int(imgName)==51 or int(imgName)==61 or int(imgName)==71):
                filename= "cat1_"+file_batch[i].split("/")[-1]
                shutil.copy(file_batch[i],os.path.join(outf+dest_dir,file_batch[i].split("/")[-1]))
                
            if (int(imgName) == 2 or int(imgName) == 12 or int(imgName) == 22 or int(imgName) == 32 or int(imgName) == 42 or int(imgName) == 52 or int(imgName) == 62 or int(imgName) == 72):
                filename = "cat2_" + file_batch[i].split("/")[-1]
            if (int(imgName) == 3 or int(imgName) == 13 or int(imgName) == 23 or int(imgName) == 33 or int(imgName) == 43 or int(imgName) == 53 or int(imgName) == 63 or int(imgName) == 73):
                filename = "cat3_" + file_batch[i].split("/")[-1]
            if (int(imgName) == 4 or int(imgName) == 14 or int(imgName) == 24 or int(imgName) == 34 or int(imgName) == 44 or int(imgName) == 54 or int(imgName) == 64 or int(imgName) == 74):
                filename = "cat4_" + file_batch[i].split("/")[-1]
            if (int(imgName) == 5 or int(imgName) == 15 or int(imgName) == 25 or int(imgName) == 35 or int(imgName) == 45 or int(imgName) == 55 or int(imgName) == 65 or int(imgName) == 75):
                filename = "cat5_" + file_batch[i].split("/")[-1]
            print(file_batch[i],os.path.join(outf+dest_dir,filename))
            cropped_img.save(os.path.join(outf+dest_dir,filename))
            y1 = y1 + 130
            y2 = y2 + 130
        x1 = x1 + 130
        x2 = x2 + 130
        y1 = 2
        y2 = 130
