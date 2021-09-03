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
#from testresult import  *
#from FaceMatcherAllFeatureExtract import  *  # Single
#from FaceMatcherAll import *
#from FaceMatcherROCMathAll import *
#from ImposterGeninePairGenerator import  *
torch.cuda.set_device('cuda:0')

layer_names = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
               'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
               'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
               'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
               'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5']
               
default_content_layers = ['relu1_1', 'relu2_1', 'relu3_1']

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='/home/praveen/Downloads/Pattern/IndiaSkin/PR_Indian_Childdata_Skincolor/',
                    help='path to dataset folder (must follow PyTorch ImageFolder structure)')
parser.add_argument('--batch_size', type=int,
                    default=32, help='input batch size, default=128')
parser.add_argument('--image_size', type=int, default=128,
                    help='height/width length of the input images, default=64')
parser.add_argument('--nz', type=int, default=50,
                    help='size of the latent vector z, default=100')
parser.add_argument('--nef', type=int, default=64,
                    help='number of output channels for the first encoder layer, default=32')
parser.add_argument('--ndf', type=int, default=64,
                    help='number of output channels for the first decoder layer, default=32')
parser.add_argument('--instance_norm', action='store_true',
                    help='use instance norm layer instead of batch norm')
parser.add_argument('--content_layers', type=str, nargs='?', default=None,
                    help='name of the layers to be used to compute the feature perceptual loss, default=[relu3_1, relu4_1, relu5_1]')
parser.add_argument('--niter', type=int, default=22001,
                    help='number of epochs to train for, default=10')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam, default=0.5')
parser.add_argument('--cuda', action='store_true',default=True, help='enables cuda')
parser.add_argument('--encoder', default='',
                    help="path to encoder (to continue training)")
parser.add_argument('--decoder', default='',
                    help="path to decoder (to continue training)")
parser.add_argument('--dimg', default='',
                    help="path to encoder (to continue training)")
parser.add_argument('--dz', default=None,
                    help="path to decoder (to continue training)")
parser.add_argument('--outf', default='/home/praveen/Downloads/Pattern/IndiaSkin/PRIndianChildGANSkinOutput/',

                    help='folder to output images and model checkpoints')
parser.add_argument('--manual_seed', type=int, help='manual seed')
parser.add_argument('--log_interval', type=int, default=1, help='number of iterations between each stdout logging, default=1')
parser.add_argument('--img_interval', type=int, default=1000, help='number of iterations between each image saving, default=100')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')
args = parser.parse_args()
print(args)
use_cuda = torch.cuda.is_available()
n_z = 50
n_l = 5
n_channel = 3
n_disc = 16
n_gen = 64
n_age = int(n_z/n_l) #12
n_gender = int(n_z/2) #25
try:
    os.makedirs(args.outf)
except OSError:
    pass

if args.manual_seed is None:
    args.manual_seed = random.randint(1, 10000)
print("Random Seed: ", args.manual_seed)
random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.manual_seed)

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Dataset loading
# Normalization mean and standard deviation are set accordingly to the ones used
# to train the vgg19 in torchvision model zoo
# https://github.com/pytorch/vision
transform = transforms.Compose([
    transforms.Scale(args.image_size),
    #transforms.CenterCrop(args.image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                         std=(0.5, 0.5, 0.5))])
datafolder = dset.ImageFolder(root=args.dataroot, transform=transform)
dataloader = torch.utils.data.DataLoader(datafolder, shuffle=True, batch_size=args.batch_size, drop_last=True)

ngpu = int(args.ngpu)
nz = int(args.nz)
nef = int(args.nef)
ndf = int(args.ndf)
nc = 3
out_size = args.image_size // 16  # 64
if args.instance_norm:
    Normalize = nn.InstanceNorm2d
else:
    Normalize = nn.BatchNorm2d
if args.content_layers is None:
    content_layers = default_content_layers
else:
    content_layers = args.content_layers

if use_cuda:
    BCE = nn.BCELoss().cuda()
    L1  = nn.L1Loss().cuda()
    CE = nn.CrossEntropyLoss().cuda()
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight.data, a=0.01)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        init.normal(m.weight.data, std=0.015)
        m.bias.data.zero_()
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
        #print("====Attamtion SRGAN",m_batchsize,C,width,height)
        proj_query = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1)
        proj_key = self.key_conv(x).view(m_batchsize,-1,width*height)
        energy = torch.bmm(proj_query,proj_key) #batch matrix-matrix product of matrices store
        attention = self.softmax(energy)
        proj_value  = self.value_conv(x).view(m_batchsize,-1,width*height)
        out  = torch.bmm(proj_value,attention.permute(0,2,1))
        out  = out.view(m_batchsize,C,width,height)
        out = self.gamma*out +x
        return out,attention



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


class VGG(nn.Module):

    def __init__(self, ngpu):
        super(VGG, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.features = nn.Sequential()
        for i, module in enumerate(features):
            name = layer_names[i]
            self.features.add_module(name, module)

    def forward(self, input):
        batch_size = input.size(0)
        all_outputs = []
        output = input
        for name, module in self.features.named_children():
            
            output = module(output)
            if name in content_layers:
                all_outputs.append(output.view(batch_size, -1))
        return all_outputs


descriptor = VGG(ngpu)



class _Encoder(nn.Module):
    def __init__(self, ngpu):
        super(_Encoder, self).__init__()
        self.ngpu = ngpu
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
        if args.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

    def forward(self, input):
        batch_size = input.size(0)
        #print("Decoder without Parallel...")
        hidden = self.encoder(input)
        hidden = self.resnet_blocks(hidden)
        hidden = self.encoder_second(hidden)
        #print("Encoder Size",hidden.size())
        out,ep1  = self.attn1(hidden)
        #print("Attan out Size", out.size())
        hidden = out.view(batch_size, -1)
        mean, logvar = self.mean(hidden), self.logvar(hidden)
        latent_z = self.sampler(mean, logvar)
        return latent_z,ep1


encoder = _Encoder(ngpu)
#encoder.apply(weights_init)
if args.encoder != '':
   encoder.load_state_dict(torch.load(args.encoder))
#print(encoder)

class _Decoder(nn.Module):
    def __init__(self, ngpu):
        super(_Decoder, self).__init__()
        self.ngpu = ngpu
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


class Dimg(nn.Module):
    def __init__(self):
        super(Dimg,self).__init__()
        self.conv_img = nn.Sequential(
            nn.Conv2d(n_channel,n_disc,4,2,1),
        )
        self.conv_l = nn.Sequential(
            nn.ConvTranspose2d(n_l*n_age+n_gender, n_l*n_age+n_gender, 64, 1, 0),
            nn.ReLU(),
            nn.InstanceNorm2d(n_l * n_age + n_gender)
        )
        self.total_conv = nn.Sequential(
            nn.Conv2d(n_disc+n_l*n_age+n_gender,n_disc*2,4,2,1),
            nn.ReLU(),
            nn.InstanceNorm2d(n_disc*2),

            nn.Conv2d(n_disc*2,n_disc*4,4,2,1),
            nn.ReLU(),
            nn.InstanceNorm2d(n_disc * 4),

            nn.Conv2d(n_disc*4,n_disc*8,4,2,1),
            nn.ReLU(),
            nn.InstanceNorm2d(n_disc * 8),

        )
        self.attn1 = Self_Attn(n_disc*8, 'relu')

        self.fc_common = nn.Sequential(
            nn.Linear(8 * 8 * args.image_size, 1024),
            nn.ReLU()
        )
        self.fc_head1 = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        self.fc_head2 = nn.Sequential(
            nn.Linear(1024, n_l),
            nn.Softmax()
        )

        ''' Patch GAN
        self.fc_head1 = nn.Sequential(
            nn.Linear(8 * 8, 64),
            nn.Softmax()
        )

        self.fc_head2 = nn.Sequential(
            nn.Linear(8*8,n_l),
            nn.Softmax()
        )
        self.patchconv  = nn.Sequential(
            nn.Conv2d(n_disc*8,1,3,1,1),

        )
 
        '''
    def forward(self,img,age,gender):
        l = age.repeat(1,n_age,1,1,)
        k = gender.repeat(1,n_gender,1,1,)
        conv_img = self.conv_img(img)
        conv_l   = self.conv_l(torch.cat([l,k],dim=1)) # torch.cat([l,k] size = 20 * 73 * 1 * 1, # size = 20 * 73 * 64 * 64
        catted   = torch.cat((conv_img,conv_l),dim=1)
        total_conv = self.total_conv(catted)
        out, dp2 = self.attn1(total_conv)
        #patchconv=   self.patchconv(out).view(-1,8*8)
        #print("After Path GAN",patchconv.size())
        total_conv = out.view(-1,8*8*args.image_size)

        body = self.fc_common(total_conv)  # size = 20  * 1024
        head1 = self.fc_head1(body)  # size = 20 * 1
        head2 = self.fc_head2(body)  # size = 20 * 4
        return head1,head2,dp2

decoder = _Decoder(ngpu)
#decoder.apply(weights_init)
if args.decoder != '':
    decoder.load_state_dict(torch.load(args.decoder))
print(decoder)

netD_img = Dimg().cuda()
if args.dimg != '':
   netD_img.load_state_dict(torch.load(args.dimg))

#netD_img.apply(weights_init)

#netD_z  = Dz().cuda()
#if args.dz != '':
#    netD_z.load_state_dict(torch.load(args.dz))
#netD_z.apply(weights_init)

mse = nn.MSELoss()
def fpl_criterion(recon_features, targets):
    fpl = 0
    for f, target in zip(recon_features, targets):
        fpl += mse(f, target.detach())#.div(f.size(1))
    return fpl

kld_criterion = nn.KLDivLoss()


input = torch.FloatTensor(
    args.batch_size, nc, args.image_size, args.image_size)
latent_labels = torch.FloatTensor(args.batch_size, nz).fill_(1)

if use_cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()
    descriptor = descriptor.cuda()
    input = input.cuda()
    latent_labels = latent_labels.cuda()

input = Variable(input)
latent_labels = Variable(latent_labels)
optimizerE = optim.Adam(encoder.parameters(),lr=0.0001,betas=(0.5,0.999))
#optimizerD_z = optim.Adam(netD_z.parameters(),lr=0.0001,betas=(0.5,0.999))
optimizerD_img = optim.Adam(netD_img.parameters(),lr=0.0002,betas=(0.5,0.999))
optimizerD = optim.Adam(decoder.parameters(),lr=0.0002,betas=(0.5,0.999))
## fixed variables to regress / progress age
fixed_l = -torch.ones(40*5).view(40,5)
for i,l in enumerate(fixed_l):
    l[i//8] = 1
fixed_l_v = Variable(fixed_l)
if use_cuda:
    fixed_l_v = fixed_l_v.cuda()
encoder.train()
decoder.train()
train_loss = 0
d_loss = []
g_loss = []
def deleteContent(pairs_file):
    pairs_file1 = open("clfpairwgancriticres9lr0001.txt", "w+")
    pairs_file1.seek(0)
    pairs_file1.truncate()
i=0
for epoch in range(args.niter):
    torch.cuda.empty_cache()
    for g_iter in range(5): # generator  5 times
        for p in netD_img.parameters():
            p.requires_grad = True
        d_loss_real = 0
        d_loss_fake = 0
        Wasserstein_D = 0
        for d_iter in range(1): # dis img 1 time
            netD_img.zero_grad()
            dataloader_iterator = iter(dataloader)
            img_data,img_label = next(dataloader_iterator)
            img_data_v = Variable(img_data)

            img_age = img_label / 2  # size = no of image * 1
            img_gender = img_label % 2 * 2 - 1  # size = no of image * 1
            img_age_v = Variable(img_age).view(-1, 1)
            img_gender_v = Variable(img_gender.float())
            if epoch == 0 and i==0:
               fixed_noise = img_data[:8].repeat(5, 1, 1, 1)  # size = 32 * 3 * 128 * 128
               fixed_g = img_gender[:8].view(-1, 1).repeat(5, 1)  # size = 32 * 1
               fixed_img_v = Variable(fixed_noise)
               fixed_g_v = Variable(fixed_g)
               pickle.dump(fixed_noise, open("fixed_noise.p", "wb"))

            if use_cuda and epoch == 0 and i==0:
                    fixed_img_v = fixed_img_v.cuda()
                    fixed_g_v = fixed_g_v.cuda()
                    vutils.save_image(fixed_img_v.data,
                                      '{}/initial_inputs.png'.format(args.outf),
                                      normalize=True)
            if use_cuda:
                    img_data_v = img_data_v.cuda()
                    img_age_v = img_age_v.cuda()
                    img_gender_v = img_gender_v.cuda()

            # make one hot encoding version of label
            batchSize = img_data_v.size(0)
            age_ohe = one_hot(img_age, batchSize, n_l, use_cuda)  # size = noOfImages * n_l

            # prior distribution z_star, real_label, fake_label
            z_star = Variable(torch.FloatTensor(batchSize * n_z).uniform_(-1, 1)).view(batchSize, n_z)
            real_label = Variable(torch.ones(batchSize).fill_(1)).view(-1, 1)
            fake_label = Variable(torch.ones(batchSize).fill_(0)).view(-1, 1)
            real_label_dim = Variable(torch.ones(batchSize, 64).fill_(1)).view(-1, 1).cuda()
            fake_label_dim = Variable(torch.ones(batchSize,64).fill_(0)).view(-1, 1).cuda()

            if use_cuda:
                z_star, real_label, fake_label = z_star.cuda(), real_label.cuda(), fake_label.cuda()

            ## train Encoder and Generator with reconstruction loss

            optimizerE.zero_grad()
            optimizerD.zero_grad()
            input.data.copy_(img_data)

            latent_z,ep1 = encoder(input)
            #targets = descriptor(input)
            #kld = kld_criterion(F.log_softmax(latent_z), latent_labels)
            #kld.backward(create_graph=True)

            recon = decoder(latent_z,age_ohe,img_gender_v)
            ## train D_img with real images
            netD_img.zero_grad()
            D_img, D_clf,dp2 = netD_img(img_data_v, age_ohe.view(batchSize, n_l, 1, 1), img_gender_v.view(batchSize, 1, 1, 1))
            #d_loss_real = torch.nn.ReLU()(1.0 - D_img).mean()
            d_loss_real = - torch.mean(D_img)
            d_loss_real.backward(retain_graph=False)
            D_reconst, _,dp2 = netD_img(recon, age_ohe.view(batchSize, n_l, 1, 1),
                                        img_gender_v.view(batchSize, 1, 1, 1))
            #d_loss_fake = torch.nn.ReLU()(1.0 + D_reconst).mean()
            d_loss_fake = D_reconst.mean()
            # Total Discriminator_img Loss (pass x+l to d_img and pass G+l in d_img)
            #D_loss = BCE(D_img, fake_label_dim) + BCE(D_reconst, fake_label_dim)
            #D_loss  =  d_loss_real  + d_loss_fake
            d_loss_fake.backward(retain_graph=True)
            #gradient_penalty = calculate_gradient_penalty(img_data.data, recon.data, age_ohe, img_gender_v)
            eta = torch.FloatTensor(args.batch_size, 1, 1, 1).uniform_(0, 1).cuda()
            eta = eta.expand(args.batch_size, img_data.size(1), img_data.size(2), img_data.size(3)).cuda()

            if use_cuda:
                eta = eta.cuda()
            else:
                eta = eta

            interpolated = eta * img_data.cuda() + ((1 - eta) * recon.cuda())
            if use_cuda:
                interpolated = interpolated.cuda()
            else:
                interpolated = interpolated

            # define it to calculate gradient
            interpolated = Variable(interpolated, requires_grad=True)
            # calculate gradient of probabilites with respect to example

            prob_interpolated,_,_ = netD_img(interpolated, age_ohe.view(batchSize, n_l, 1, 1),
                                         img_gender_v.view(batchSize, 1, 1, 1))
            # calcualte gradient of probabilities with respect to examples
            gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                      grad_outputs=torch.ones(prob_interpolated.size()).cuda(
                                      ) if use_cuda else torch.ones(
                                          prob_interpolated.size()),
                                      create_graph=True, retain_graph=True)[0]
            grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10

            grad_penalty.backward()
            d_loss = d_loss_real - d_loss_fake + grad_penalty
            Wasserstein_D = d_loss_real - d_loss_fake
            optimizerD_img.step()

        # Generator update
        for p in netD_img.parameters():
            p.requires_grad=False

        #encoder.zero_grad()
        #decoder.zero_grad()
        latent_z, ep1 = encoder(input)
        targets = descriptor(input)
        kld = kld_criterion(F.log_softmax(latent_z), latent_labels)
        kld.backward(create_graph=True)
        recon = decoder(latent_z, age_ohe, img_gender_v)
        recon_features = descriptor(recon)
        fpl = fpl_criterion(recon_features, targets)
        fpl.backward()
        #Dz = netD_z(latent_z)
        # send z to d_z
        #Ez_loss = BCE(Dz, real_label)
        loss = kld + fpl
        train_loss += loss.item()
        D_reconst,_,dp2 = netD_img(recon, age_ohe.view(batchSize, n_l, 1, 1), img_gender_v.view(batchSize, 1, 1, 1))
        G_img_loss = - D_reconst.mean()  # WGAN-gp
        reconst = decoder(latent_z.detach(), age_ohe, img_gender_v)
        G_tv_loss = TV_LOSS(reconst)
        #EG_loss = loss + 0.0001 * G_img_loss + 0.01 * Ez_loss + G_tv_loss
        EG_loss = loss + 0.0001 * G_img_loss +  G_tv_loss
        #EG_loss.backward()
        optimizerE.step()
        optimizerD.step()
    #     ## train netD_z with prior distribution U(-1,1)
        #netD_z.zero_grad()
        #Dz_prior = netD_z(z_star)
        #Dz = netD_z(latent_z.detach())
        # Total Discriminator_z Loss (pass z to d_z and pass z_prior in d_z)
        #Dz_loss = BCE(Dz_prior, real_label) + BCE(Dz, fake_label)
        #Dz_loss.backward()
        #optimizerD_z.step()
        ## train D_img with real images
        netD_img.zero_grad()
        if i == 0:
            vutils.save_image(input.data,
                              '{}/inputs.png'.format(args.outf),
                              normalize=True)
    fixed_z,ep2 = encoder(fixed_img_v)
    fixed_fake = decoder(fixed_z, fixed_l_v, fixed_g_v)
    if epoch%3000==0:
        vutils.save_image(fixed_fake.data,
                      '%s/reconst_epoch%03d.png' % (args.outf, epoch + 1),
                      normalize=True)

    # do checkpointing
    
    if epoch % 3000 == 0:
        torch.save(encoder.state_dict(), '{}/encoder_epoch_{}.pth'.format(args.outf, epoch))
        torch.save(decoder.state_dict(), '{}/decoder_epoch_{}.pth'.format(args.outf, epoch))
        #torch.save(netD_z.state_dict(), '{}/dz_epoch_{}.pth'.format(args.outf, epoch))
        torch.save(netD_img.state_dict(), '{}/dimag_epoch_{}.pth'.format(args.outf, epoch))
        msg1 = "epoch:{}, step:{}".format(epoch + 1, i + 1)
        msg2 = format("FPL loss :%f" % (fpl.item()), "<30") + "|" + format("KLD :%f" % (kld.item()), "<30")
        msg3 = format("G_img_loss:%f" % (G_img_loss.item()), "<30")
        msg4 = format("G_tv_loss:%f" % (G_tv_loss.item()), "<30") #+ "|" + "Ez_loss:%f" % (Ez_loss.item())
        msg5 = format("D_img:%f" % (D_img.mean().item()), "<30") + "|" + format(
            "D_reconst:%f" % (D_reconst.mean().item()), "<30") \
               + "|" + format("D_loss:%f" % (d_loss.item()), "<30")
        #msg6 = format("D_z:%f" % (Dz.mean().item()), "<30") + "|" + format("D_z_prior:%f" % (Dz_prior.mean().item()),
        #                                                                    "<30") \
        #       + "|" + format("Dz_loss:%f" % (Dz_loss.item()), "<30")

        print()
        print(msg1)
        print(msg2)
        print(msg3)
        print(msg4)
        print(msg5)
        #print(msg6)
        print()
        print("-" * 80)
        #evaluate_result(epoch)
        #pairs_file = imposter_genuine()
        #feature_extract(epoch)
        #deleteContent(pairs_file)
        print("================="+str(epoch)+"====Complet===========")
        #matcher()
    i += 1
