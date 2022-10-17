#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import cv2
import numpy as np
import torch 
import timeit
#import onnxruntime
from torch.nn import functional as F
from torchvision.transforms.functional import normalize
from torch import nn
import math
from collections import OrderedDict
from noise_main import noise_dict

class ResBlock(nn.Module):
    """Residual block with upsampling/downsampling.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
    """

    def __init__(self, in_channels, out_channels, mode='down'):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        if mode == 'down':
            self.scale_factor = 0.5
        elif mode == 'up':
            self.scale_factor = 2

    def forward(self, x):
        out = F.leaky_relu_(self.conv1(x), negative_slope=0.2)
        # upsample/downsample
        out = F.interpolate(out, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        out = F.leaky_relu_(self.conv2(out), negative_slope=0.2)
        # skip
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        skip = self.skip(x)
        out = out + skip
        return out


class ConstantInput(nn.Module):
    """Constant input.

    Args:
        num_channel (int): Channel number of constant input.
        size (int): Spatial size of constant input.
    """

    def __init__(self, num_channel, size):
        super(ConstantInput, self).__init__()
        self.weight = nn.Parameter(torch.randn(1, num_channel, size, size)) # [1, 512, 4, 4]
    
    def forward(self, batch):
        out = self.weight.repeat(batch, 1, 1, 1)
        
        return out

class GFPGAN(nn.Module):
    def __init__(self):
        super(GFPGAN, self).__init__()
        unet_narrow = 0.5
        channel_multiplier=2
        channels = {
            '4': int(512 * unet_narrow),
            '8': int(512 * unet_narrow),
            '16': int(512 * unet_narrow),
            '32': int(512 * unet_narrow),
            '64': int(256 * channel_multiplier * unet_narrow),
            '128': int(128 * channel_multiplier * unet_narrow),
            '256': int(64 * channel_multiplier * unet_narrow),
            '512': int(32 * channel_multiplier * unet_narrow),
            '1024': int(16 * channel_multiplier * unet_narrow)
        }

        self.conv_body_first = nn.Conv2d(3, 32, 1)
        self.conv_body_down = nn.ModuleList()
        
        in_channels = channels['512']
        for i in range(9, 2, -1):
            out_channels = channels[f'{2**(i - 1)}']
            self.conv_body_down.append(ResBlock(in_channels, out_channels, mode='down'))
            in_channels = out_channels
        num_style_feat = 512
        self.final_conv = nn.Conv2d(in_channels, channels['4'], 3, 1, 1)
        linear_out_channel = (int(math.log(512, 2)) * 2 - 2) * num_style_feat
        self.final_linear = nn.Linear(channels['4'] * 4 * 4, linear_out_channel)
        
        # upsample
        in_channels = channels['4']
        self.conv_body_up = nn.ModuleList()
        for i in range(3, 9 + 1):
            out_channels = channels[f'{2**i}']
            self.conv_body_up.append(ResBlock(in_channels, out_channels, mode='up'))
            in_channels = out_channels

        # for SFT
        self.condition_scale = nn.ModuleList()
        self.condition_shift = nn.ModuleList()
        for i in range(3, 9 + 1):
            out_channels = channels[f'{2**i}']
            sft_out_channels = out_channels
            self.condition_scale.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1), nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_channels, sft_out_channels, 3, 1, 1)))
            self.condition_shift.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1), nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_channels, sft_out_channels, 3, 1, 1)))
        
        self.stylegan_decoderdotconstant_input = ConstantInput(512, size=4)
        
        
        # self.style_conv1
        self.stylegan_decoderdotstyle_conv1dotmodulated_convdotmodulation = nn.Linear(512, 512, bias=True)
        
        self.stylegan_decoderdotstyle_conv1dotmodulated_convdotweight = nn.Parameter(
            torch.randn(1, 512, 512, 3, 3) /
            math.sqrt(512 * 3**2))
        
        self.stylegan_decoderdotstyle_conv1dotweight = nn.Parameter(torch.zeros(1))  # for noise injection
        
        
        self.stylegan_decoderdotstyle_conv1dotbias = nn.Parameter(torch.zeros(1, 512, 1, 1))
        self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        
        # toRGB
        self.stylegan_decoderdotto_rgb1dotmodulated_convdotmodulation = nn.Linear(512, 512, bias=True)
        self.stylegan_decoderdotto_rgb1dotmodulated_convdotweight = nn.Parameter(
            torch.randn(1, 3, 512, 1, 1) /
            math.sqrt(512 * 1**2))
        self.stylegan_decoderdotto_rgb1dotbias = nn.Parameter(torch.zeros(1, 3, 1, 1))
        
        # i = 1
        self.stylegan_decoderdotstyle_convsdot0dotmodulated_convdotmodulation = nn.Linear(512, 512, bias=True)
        self.stylegan_decoderdotstyle_convsdot0dotmodulated_convdotweight = nn.Parameter(
            torch.randn(1, 512, 512, 3, 3) /
            math.sqrt(512 * 3**2))
        self.stylegan_decoderdotstyle_convsdot0dotweight = nn.Parameter(torch.zeros(1))  # for noise injection
        self.stylegan_decoderdotstyle_convsdot0dotbias = nn.Parameter(torch.zeros(1, 512, 1, 1))
        self.stylegan_decoderdotstyle_convsdot1dotmodulated_convdotmodulation = nn.Linear(512, 512, bias=True)
        self.stylegan_decoderdotstyle_convsdot1dotmodulated_convdotweight = nn.Parameter(
            torch.randn(1, 512, 512, 3, 3) /
            math.sqrt(512 * 3**2))
        self.stylegan_decoderdotstyle_convsdot1dotweight = nn.Parameter(torch.zeros(1))  
        self.stylegan_decoderdotstyle_convsdot1dotbias = nn.Parameter(torch.zeros(1, 512, 1, 1))
        #self.stylegan_decoderdotstyle_convsdot0dotmodulated_convdotmodulation = nn.Linear(512, 512, bias=True)
        self.stylegan_decoderdotto_rgbsdot0dotmodulated_convdotmodulation = nn.Linear(512, 512, bias=True)
        self.stylegan_decoderdotto_rgbsdot0dotmodulated_convdotweight = nn.Parameter(
            torch.randn(1, 3, 512, 1, 1) /
            math.sqrt(512 * 1**2))
        self.stylegan_decoderdotto_rgbsdot0dotbias = nn.Parameter(torch.zeros(1, 3, 1, 1))

        #i = 3
        self.stylegan_decoderdotstyle_convsdot2dotmodulated_convdotmodulation = nn.Linear(512, 512, bias=True)
        self.stylegan_decoderdotstyle_convsdot2dotmodulated_convdotweight = nn.Parameter(
            torch.randn(1, 512, 512, 3, 3) /
            math.sqrt(512 * 3**2))
        self.stylegan_decoderdotstyle_convsdot2dotweight = nn.Parameter(torch.zeros(1))  # for noise injection
        self.stylegan_decoderdotstyle_convsdot2dotbias = nn.Parameter(torch.zeros(1, 512, 1, 1))
        self.stylegan_decoderdotstyle_convsdot3dotmodulated_convdotmodulation = nn.Linear(512, 512, bias=True)
        self.stylegan_decoderdotstyle_convsdot3dotmodulated_convdotweight = nn.Parameter(
            torch.randn(1, 512, 512, 3, 3) /
            math.sqrt(512 * 3**2))
        self.stylegan_decoderdotstyle_convsdot3dotweight = nn.Parameter(torch.zeros(1))  
        self.stylegan_decoderdotstyle_convsdot3dotbias = nn.Parameter(torch.zeros(1, 512, 1, 1))
        self.stylegan_decoderdotstyle_convsdot2dotmodulated_convdotmodulation = nn.Linear(512, 512, bias=True)
        self.stylegan_decoderdotto_rgbsdot1dotmodulated_convdotmodulation = nn.Linear(512, 512, bias=True)
        self.stylegan_decoderdotto_rgbsdot1dotmodulated_convdotweight = nn.Parameter(
            torch.randn(1, 3, 512, 1, 1) /
            math.sqrt(512 * 1**2))
        self.stylegan_decoderdotto_rgbsdot1dotbias = nn.Parameter(torch.zeros(1, 3, 1, 1))
        
        #i = 5
        self.stylegan_decoderdotstyle_convsdot4dotmodulated_convdotmodulation = nn.Linear(512, 512, bias=True)
        self.stylegan_decoderdotstyle_convsdot4dotmodulated_convdotweight = nn.Parameter(
            torch.randn(1, 512, 512, 3, 3) /
            math.sqrt(512 * 3**2))
        self.stylegan_decoderdotstyle_convsdot4dotweight = nn.Parameter(torch.zeros(1))  # for noise injection
        self.stylegan_decoderdotstyle_convsdot4dotbias = nn.Parameter(torch.zeros(1, 512, 1, 1))
        self.stylegan_decoderdotstyle_convsdot5dotmodulated_convdotmodulation = nn.Linear(512, 512, bias=True)
        self.stylegan_decoderdotstyle_convsdot5dotmodulated_convdotweight = nn.Parameter(
            torch.randn(1, 512, 512, 3, 3) /
            math.sqrt(512 * 3**2))
        self.stylegan_decoderdotstyle_convsdot5dotweight = nn.Parameter(torch.zeros(1))  
        self.stylegan_decoderdotstyle_convsdot5dotbias = nn.Parameter(torch.zeros(1, 512, 1, 1))
        self.stylegan_decoderdotto_rgbsdot2dotmodulated_convdotmodulation = nn.Linear(512, 512, bias=True)
        self.stylegan_decoderdotto_rgbsdot2dotmodulated_convdotweight = nn.Parameter(
            torch.randn(1, 3, 512, 1, 1) /
            math.sqrt(512 * 1**2))
        self.stylegan_decoderdotto_rgbsdot2dotbias = nn.Parameter(torch.zeros(1, 3, 1, 1))
        
        #i = 7
        self.stylegan_decoderdotstyle_convsdot6dotmodulated_convdotmodulation = nn.Linear(512, 512, bias=True)
        self.stylegan_decoderdotstyle_convsdot6dotmodulated_convdotweight = nn.Parameter(
            torch.randn(1, 512, 512, 3, 3) /
            math.sqrt(512 * 3**2))
        self.stylegan_decoderdotstyle_convsdot6dotweight = nn.Parameter(torch.zeros(1))  # for noise injection
        self.stylegan_decoderdotstyle_convsdot6dotbias = nn.Parameter(torch.zeros(1, 512, 1, 1))
        self.stylegan_decoderdotstyle_convsdot7dotmodulated_convdotmodulation = nn.Linear(512, 512, bias=True)
        self.stylegan_decoderdotstyle_convsdot7dotmodulated_convdotweight = nn.Parameter(
            torch.randn(1, 512, 512, 3, 3) /
            math.sqrt(512 * 3**2))
        self.stylegan_decoderdotstyle_convsdot7dotweight = nn.Parameter(torch.zeros(1))  
        self.stylegan_decoderdotstyle_convsdot7dotbias = nn.Parameter(torch.zeros(1, 512, 1, 1))
        
        self.stylegan_decoderdotto_rgbsdot3dotmodulated_convdotmodulation = nn.Linear(512, 512, bias=True)
        self.stylegan_decoderdotto_rgbsdot3dotmodulated_convdotweight = nn.Parameter(
            torch.randn(1, 3, 512, 1, 1) /
            math.sqrt(512 * 1**2))
        self.stylegan_decoderdotto_rgbsdot3dotbias = nn.Parameter(torch.zeros(1, 3, 1, 1))

        #i = 9
        self.stylegan_decoderdotstyle_convsdot8dotmodulated_convdotmodulation = nn.Linear(512, 512, bias=True)
        self.stylegan_decoderdotstyle_convsdot8dotmodulated_convdotweight = nn.Parameter(
            torch.randn(1, 256, 512, 3, 3) /
            math.sqrt(256 * 3**2))
        self.stylegan_decoderdotstyle_convsdot8dotweight = nn.Parameter(torch.zeros(1))  # for noise injection
        self.stylegan_decoderdotstyle_convsdot8dotbias = nn.Parameter(torch.zeros(1, 256, 1, 1))
        
        self.stylegan_decoderdotstyle_convsdot9dotmodulated_convdotmodulation = nn.Linear(512, 256, bias=True)
        self.stylegan_decoderdotstyle_convsdot9dotmodulated_convdotweight = nn.Parameter(
            torch.randn(1, 256, 256, 3, 3) /
            math.sqrt(256 * 3**2))
        
        self.stylegan_decoderdotstyle_convsdot9dotweight = nn.Parameter(torch.zeros(1))  
        self.stylegan_decoderdotstyle_convsdot9dotbias = nn.Parameter(torch.zeros(1, 256, 1, 1))
        self.stylegan_decoderdotto_rgbsdot4dotmodulated_convdotmodulation = nn.Linear(512, 256, bias=True)
        self.stylegan_decoderdotto_rgbsdot4dotmodulated_convdotweight = nn.Parameter(
            torch.randn(1, 3, 256, 1, 1) /
            math.sqrt(256 * 1**2))
        self.stylegan_decoderdotto_rgbsdot4dotbias = nn.Parameter(torch.zeros(1, 3, 1, 1))
        
        #i = 11
        self.stylegan_decoderdotstyle_convsdot10dotmodulated_convdotmodulation = nn.Linear(512, 256, bias=True)
        self.stylegan_decoderdotstyle_convsdot10dotmodulated_convdotweight = nn.Parameter(
            torch.randn(1, 128, 256, 3, 3) /
            math.sqrt(128 * 3**2))
        self.stylegan_decoderdotstyle_convsdot10dotweight = nn.Parameter(torch.zeros(1))  # for noise injection
        self.stylegan_decoderdotstyle_convsdot10dotbias = nn.Parameter(torch.zeros(1, 128, 1, 1))
        self.stylegan_decoderdotstyle_convsdot11dotmodulated_convdotmodulation = nn.Linear(512, 128, bias=True)
        
        self.stylegan_decoderdotstyle_convsdot11dotmodulated_convdotweight = nn.Parameter(
            torch.randn(1, 128, 128, 3, 3) /
            math.sqrt(128 * 3**2)) 
        self.stylegan_decoderdotstyle_convsdot11dotweight = nn.Parameter(torch.zeros(1))  
        self.stylegan_decoderdotstyle_convsdot11dotbias = nn.Parameter(torch.zeros(1, 128, 1, 1))
        self.stylegan_decoderdotto_rgbsdot5dotmodulated_convdotmodulation = nn.Linear(512, 128, bias=True)
        self.stylegan_decoderdotto_rgbsdot5dotmodulated_convdotweight = nn.Parameter(
            torch.randn(1, 3, 128, 1, 1) /
            math.sqrt(128 * 1**2))
        self.stylegan_decoderdotto_rgbsdot5dotbias = nn.Parameter(torch.zeros(1, 3, 1, 1))
        
        #i = 13
        self.stylegan_decoderdotstyle_convsdot12dotmodulated_convdotmodulation = nn.Linear(512, 128, bias=True)
        self.stylegan_decoderdotstyle_convsdot12dotmodulated_convdotweight = nn.Parameter(
            torch.randn(1, 64, 128, 3, 3) /
            math.sqrt(64 * 3**2))
        self.stylegan_decoderdotstyle_convsdot12dotweight = nn.Parameter(torch.zeros(1))  # for noise injection
        self.stylegan_decoderdotstyle_convsdot12dotbias = nn.Parameter(torch.zeros(1, 64, 1, 1))
        self.stylegan_decoderdotstyle_convsdot13dotmodulated_convdotmodulation = nn.Linear(512, 64, bias=True)
        self.stylegan_decoderdotstyle_convsdot13dotmodulated_convdotweight = nn.Parameter(
            torch.randn(1, 64, 64, 3, 3) /
            math.sqrt(64 * 3**2))
        self.stylegan_decoderdotstyle_convsdot13dotweight = nn.Parameter(torch.zeros(1))  
        self.stylegan_decoderdotstyle_convsdot13dotbias = nn.Parameter(torch.zeros(1, 64, 1, 1))
        self.stylegan_decoderdotto_rgbsdot6dotmodulated_convdotmodulation = nn.Linear(512, 64, bias=True)
        self.stylegan_decoderdotto_rgbsdot6dotmodulated_convdotweight = nn.Parameter(
            torch.randn(1, 3, 64, 1, 1) /
            math.sqrt(64 * 1**2))
        self.stylegan_decoderdotto_rgbsdot6dotbias = nn.Parameter(torch.zeros(1, 3, 1, 1))
        ''' 
        '''
    def forward(self, x):
        # encoder
        feat = F.leaky_relu_(self.conv_body_first(x), negative_slope=0.2)
        conditions = []
        unet_skips = []
        out_rgbs = []

        for i in range(7):
            feat = self.conv_body_down[i](feat)
            unet_skips.insert(0, feat)
        
        feat = F.leaky_relu_(self.final_conv(feat), negative_slope=0.2)
        
        # style code
        style_code = self.final_linear(feat.view(feat.size(0), -1))
        style_code = style_code.view(style_code.size(0), -1, 512)

        # decode
        for i in range(7):
            # add unet skip
            feat = feat + unet_skips[i]
            # ResUpLayer
            feat = self.conv_body_up[i](feat)
            # generate scale and shift for SFT layer
            scale = self.condition_scale[i](feat)
            conditions.append(scale.clone())
            shift = self.condition_shift[i](feat)
            conditions.append(shift.clone())

        styles = [style_code]

       
        #noise = [None] * 15  # for each style conv layer
        latent = styles[0]    
        out = self.stylegan_decoderdotconstant_input(latent.shape[0])
    
        b, c, h, w = 1, 512, 4, 4
        # weight modulation
        style = self.stylegan_decoderdotstyle_conv1dotmodulated_convdotmodulation(latent[:, 0]).view(b, 1, c, 1, 1)
        weight = self.stylegan_decoderdotstyle_conv1dotmodulated_convdotweight * style  # (b, c_out, c_in, k, k)
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
        weight = weight * demod.view(b, 512, 1, 1, 1)
        weight = weight.view(b * 512, c, 3, 3)
        b, c, h, w = 1, 512, 4, 4
        out = out.view(1, b * c, h, w)
        # weight: (b*c_out, c_in, k, k), groups=b
        out = F.conv2d(out, weight, padding=1, groups=b)
        out = out.view(b, 512, *out.shape[2:4]) * 2**0.5 
        b, _, h, w = 1, 512, 4, 4
        noise = noise_dict[w]
        out = out + self.stylegan_decoderdotstyle_conv1dotweight * noise
        out = out + self.stylegan_decoderdotstyle_conv1dotbias
        out = self.activate(out)
        out0 = out 
        
       
        # toRGB    
        x = out    ########
        style = latent[:, 1] ###########
        style = self.stylegan_decoderdotto_rgb1dotmodulated_convdotmodulation(latent[:, 1]).view(b, 1, c, 1, 1)
        weight = self.stylegan_decoderdotto_rgb1dotmodulated_convdotweight * style     
        weight = weight.view(3, 512, 1, 1)
        b, c, h, w = 1, 512, 4, 4
        x = x.view(1, 512, 4, 4)
        out = F.conv2d(x, weight, padding=0, groups=b)
        out = out.view(1, 3, 4, 4)
        out = out + self.stylegan_decoderdotto_rgb1dotbias
        skip = out
        out = out0
  
        # i = 1
        i = 1
        x = out
        b, c, h, w = 1, 512, 4, 4
        
        #conv1
        style = self.stylegan_decoderdotstyle_convsdot0dotmodulated_convdotmodulation(latent[:, i]).view(b, 1, c, 1, 1)
        weight = self.stylegan_decoderdotstyle_convsdot0dotmodulated_convdotweight * style
        # self.demodulate = True:
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-08)
        weight = weight * demod.view(b, 512, 1, 1, 1)
        #
        weight = weight.view(b * 512, c, 3, 3)
        # self.sample_mode == 'upsample'
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        b, c, h, w = x.shape
        x = x.view(1, b * c, h, w)
        out = F.conv2d(x, weight, padding=1, groups=b)
        out = out.view(1, 512, 8, 8) * 2 ** 0.5 
        b, _, h, w = 1,_,8,8
        noise = noise_dict[w]
        out = out + self.stylegan_decoderdotstyle_convsdot0dotweight * noise
        out = out + self.stylegan_decoderdotstyle_convsdot0dotbias
        out = self.activate(out)
        out_same, out_sft = torch.split(out, int(out.size(1) // 2), dim=1)
        out_sft = out_sft * conditions[i - 1] + conditions[i]
        out = torch.cat([out_same, out_sft], dim=1)
        #conv2
        style = self.stylegan_decoderdotstyle_convsdot1dotmodulated_convdotmodulation(latent[:, i + 1]).view(1, 1, 512, 1, 1)
        weight = self.stylegan_decoderdotstyle_convsdot1dotmodulated_convdotweight * style
        # self.demodulate = True:
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-08)
        weight = weight * demod.view(b, 512, 1, 1, 1)
        weight = weight.view(b * 512, 512, 3, 3)
        out = F.conv2d(out, weight, padding=1, groups=b)
        out = out.view(1, 512, 8, 8) * 2 ** 0.5 
        noise = noise_dict[w]
        out = out + self.stylegan_decoderdotstyle_convsdot1dotweight * noise
        out = out + self.stylegan_decoderdotstyle_convsdot1dotbias
        out = self.activate(out)
        out0 = out
        #to_rgb
        x = out
        style = latent[:, i + 2]  
        style = self.stylegan_decoderdotto_rgbsdot0dotmodulated_convdotmodulation(style).view(1, 1, 512, 1, 1)
        weight = self.stylegan_decoderdotto_rgbsdot0dotmodulated_convdotweight * style     
        weight = weight.view(3, 512, 1, 1)
        #b, c, h, w = x.shape
        x = x.view(1, b * c, h, w)
        out = F.conv2d(x, weight, padding=0, groups=b)
        out = out.view(1, 3, 8, 8)
        out = out + self.stylegan_decoderdotto_rgbsdot0dotbias
        
        skip = F.interpolate(skip, scale_factor=2, mode='bilinear', align_corners=False)
        skip = out + skip
        
        # i = 3
        out = out0
        x = out
        b, c, h, w = 1, 512, 8, 8
        i += 2
        style = latent[:, i]
        #conv1
        style = self.stylegan_decoderdotstyle_convsdot2dotmodulated_convdotmodulation(latent[:, i]).view(b, 1, c, 1, 1)
        weight = self.stylegan_decoderdotstyle_convsdot2dotmodulated_convdotweight * style
        # self.demodulate = True:
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-08)
        weight = weight * demod.view(b, 512, 1, 1, 1)
        # self.sample_mode == 'upsample'
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weight = weight.view(b * 512, c, 3, 3)
        b, c, h, w = x.shape
        x = x.view(1, b * c, h, w)
        out = F.conv2d(x, weight, padding=1, groups=b)
        out = out.view(1, 512, 16, 16) * 2 ** 0.5 
        b, _, h, w = 1, _, 16, 16
        noise = noise_dict[w]
        out = out + self.stylegan_decoderdotstyle_convsdot2dotweight * noise
        out = out + self.stylegan_decoderdotstyle_convsdot2dotbias
        out = self.activate(out)
        out_same, out_sft = torch.split(out, int(out.size(1) // 2), dim=1)
        out_sft = out_sft * conditions[i - 1] + conditions[i]
        out = torch.cat([out_same, out_sft], dim=1)
        #conv2
        style = self.stylegan_decoderdotstyle_convsdot3dotmodulated_convdotmodulation(latent[:, i + 1]).view(1, 1, 512, 1, 1)
        weight = self.stylegan_decoderdotstyle_convsdot3dotmodulated_convdotweight * style
        # self.demodulate = True:
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-08)
        weight = weight * demod.view(1, 512, 1, 1, 1)
        weight = weight.view(b * 512, 512, 3, 3)
        out = F.conv2d(out, weight, padding=1, groups=b)
        out = out.view(1, 512, 16, 16) * 2 ** 0.5 
        noise = noise_dict[w]
        out = out + self.stylegan_decoderdotstyle_convsdot3dotweight * noise
        out = out + self.stylegan_decoderdotstyle_convsdot3dotbias
        out = self.activate(out)
        out0 = out
        #to_rgb
        x = out
        style = latent[:, i + 2]  
        style = self.stylegan_decoderdotto_rgbsdot1dotmodulated_convdotmodulation(style).view(1, 1, 512, 1, 1)
        weight = self.stylegan_decoderdotto_rgbsdot1dotmodulated_convdotweight * style     
        weight = weight.view(3, 512, 1, 1)
        #b, c, h, w = x.shape
        x = x.view(1, b * c, h, w)
        out = F.conv2d(x, weight, padding=0, groups=b)
        out = out.view(1, 3, 16, 16)
        out = out + self.stylegan_decoderdotto_rgbsdot1dotbias
        skip = F.interpolate(skip, scale_factor=2, mode='bilinear', align_corners=False)
        skip = out + skip
        
        
        # i = 5
        out = out0
        x = out
        b, c, h, w = 1, 512, 32, 32
        i += 2
        style = latent[:, i] 
        #conv1
        style = self.stylegan_decoderdotstyle_convsdot4dotmodulated_convdotmodulation(latent[:, i]).view(b, 1, c, 1, 1)   
        weight = self.stylegan_decoderdotstyle_convsdot4dotmodulated_convdotweight * style
        # self.demodulate = True:
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-08)
        weight = weight * demod.view(b, 512, 1, 1, 1)
        # self.sample_mode == 'upsample'
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weight = weight.view(b * 512, c, 3, 3)    
        b, c, h, w = x.shape
        x = x.view(1, b * c, h, w)
        out = F.conv2d(x, weight, padding=1, groups=b)
        out = out.view(1, 512, 32, 32) * 2 ** 0.5 
        b, _, h, w = 1, _, 32, 32
        noise = noise_dict[w]
        out = out + self.stylegan_decoderdotstyle_convsdot4dotweight * noise
        out = out + self.stylegan_decoderdotstyle_convsdot4dotbias
        out = self.activate(out)
        
        out_same, out_sft = torch.split(out, int(out.size(1) // 2), dim=1)
        out_sft = out_sft * conditions[i - 1] + conditions[i]
        out = torch.cat([out_same, out_sft], dim=1)
       
        #conv2
        style = self.stylegan_decoderdotstyle_convsdot5dotmodulated_convdotmodulation(latent[:, i + 1]).view(1, 1, 512, 1, 1)
        weight = self.stylegan_decoderdotstyle_convsdot5dotmodulated_convdotweight * style
        # self.demodulate = True:
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-08)
        weight = weight * demod.view(1, 512, 1, 1, 1)
        weight = weight.view(b * 512, 512, 3, 3)
        out = F.conv2d(out, weight, padding=1, groups=b)
        out = out.view(1, 512, 32, 32) * 2 ** 0.5 
        noise = noise_dict[w]
        out = out + self.stylegan_decoderdotstyle_convsdot5dotweight * noise
        out = out + self.stylegan_decoderdotstyle_convsdot5dotbias
        out = self.activate(out)
        out0 = out
        #to_rgb
        x = out
        style = latent[:, i + 2]
        style = self.stylegan_decoderdotto_rgbsdot2dotmodulated_convdotmodulation(style).view(1, 1, 512, 1, 1)
        weight = self.stylegan_decoderdotto_rgbsdot2dotmodulated_convdotweight * style     
        
        weight = weight.view(3, 512, 1, 1)
        #b, c, h, w = x.shape
        x = x.view(1, b * c, h, w)
        out = F.conv2d(x, weight, padding=0, groups=b)
        
        out = out.view(1, 3, 32, 32)
        out = out + self.stylegan_decoderdotto_rgbsdot2dotbias
        skip = F.interpolate(skip, scale_factor=2, mode='bilinear', align_corners=False)
        skip = out + skip

        # i = 7
        out = out0
        x = out
        b, c, h, w = 1, 512, 32, 32
        i += 2
        style = latent[:, i]   # 数值一致
        #conv1
        style = self.stylegan_decoderdotstyle_convsdot6dotmodulated_convdotmodulation(latent[:, i]).view(b, 1, c, 1, 1)   
        weight = self.stylegan_decoderdotstyle_convsdot6dotmodulated_convdotweight * style
        # self.demodulate = True:
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-08)
        weight = weight * demod.view(b, 512, 1, 1, 1)
        # self.sample_mode == 'upsample'
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weight = weight.view(b * 512, c, 3, 3)    
        b, c, h, w = x.shape
        x = x.view(1, b * c, h, w)
        out = F.conv2d(x, weight, padding=1, groups=b)
        out = out.view(1, 512, 64, 64) * 2 ** 0.5 
        b, _, h, w = 1, _, 64, 64
        noise = noise_dict[w]
        out = out + self.stylegan_decoderdotstyle_convsdot7dotweight * noise
        out = out + self.stylegan_decoderdotstyle_convsdot7dotbias
        out = self.activate(out)
        out_same, out_sft = torch.split(out, int(out.size(1) // 2), dim=1)
        out_sft = out_sft * conditions[i - 1] + conditions[i]
        out = torch.cat([out_same, out_sft], dim=1)
        #conv2
        style = self.stylegan_decoderdotstyle_convsdot7dotmodulated_convdotmodulation(latent[:, i + 1]).view(1, 1, 512, 1, 1)
        weight = self.stylegan_decoderdotstyle_convsdot7dotmodulated_convdotweight * style
        # self.demodulate = True:
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-08)
        weight = weight * demod.view(1, 512, 1, 1, 1)
        weight = weight.view(b * 512, 512, 3, 3)
        out = F.conv2d(out, weight, padding=1, groups=b)
        out = out.view(1, 512, 64, 64) * 2 ** 0.5 
        noise = noise_dict[w]
        out = out + self.stylegan_decoderdotstyle_convsdot7dotweight * noise
        out = out + self.stylegan_decoderdotstyle_convsdot7dotbias
        out = self.activate(out)
        out0 = out
        #to_rgb
        x = out
        style = latent[:, i + 2]
        style = self.stylegan_decoderdotto_rgbsdot3dotmodulated_convdotmodulation(style).view(1, 1, 512, 1, 1)
        weight = self.stylegan_decoderdotto_rgbsdot3dotmodulated_convdotweight * style     
        weight = weight.view(3, 512, 1, 1)
        #b, c, h, w = x.shape
        x = x.view(1, b * c, h, w)
        out = F.conv2d(x, weight, padding=0, groups=b)
        out = out.view(1, 3, 64, 64)
        out = out + self.stylegan_decoderdotto_rgbsdot3dotbias
        skip = F.interpolate(skip, scale_factor=2, mode='bilinear', align_corners=False)
        skip = out + skip
        
        # i = 9
        out = out0
        x = out
        b, c, h, w = 1, 512, 64, 64
        i += 2
        style = latent[:, i]   # 数值一致
        #conv1
        style = self.stylegan_decoderdotstyle_convsdot8dotmodulated_convdotmodulation(latent[:, i]).view(b, 1, c, 1, 1)   
        weight = self.stylegan_decoderdotstyle_convsdot8dotmodulated_convdotweight * style    
        # self.demodulate = True:
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-08)
        weight = weight * demod.view(b, 256, 1, 1, 1)
        # self.sample_mode == 'upsample'
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weight = weight.view(b * 256, c, 3, 3)    
        b, c, h, w = x.shape
        x = x.view(1, b * c, h, w)
        out = F.conv2d(x, weight, padding=1, groups=b)
        out = out.view(1, 256, 128, 128) * 2 ** 0.5 
        b, _, h, w = 1, _, 128, 128
        noise = noise_dict[w]
        out = out + self.stylegan_decoderdotstyle_convsdot8dotweight * noise
        out = out + self.stylegan_decoderdotstyle_convsdot8dotbias
        out = self.activate(out)
        out_same, out_sft = torch.split(out, int(out.size(1) // 2), dim=1)
        out_sft = out_sft * conditions[i - 1] + conditions[i]
        out = torch.cat([out_same, out_sft], dim=1)
        #conv2
        style = self.stylegan_decoderdotstyle_convsdot9dotmodulated_convdotmodulation(latent[:, i + 1]).view(1, 1, 256, 1, 1)
        weight = self.stylegan_decoderdotstyle_convsdot9dotmodulated_convdotweight * style
        # self.demodulate = True:
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-08)
        weight = weight * demod.view(1, 256, 1, 1, 1)
        weight = weight.view(b * 256, 256, 3, 3)
        out = F.conv2d(out, weight, padding=1, groups=b)
        out = out.view(1, 256, 128, 128) * 2 ** 0.5 
        noise = noise_dict[w]
        out = out + self.stylegan_decoderdotstyle_convsdot9dotweight * noise
        out = out + self.stylegan_decoderdotstyle_convsdot9dotbias
        out = self.activate(out)
        out0 = out
        #to_rgb
        x = out
        style = latent[:, i + 2]
        style = self.stylegan_decoderdotto_rgbsdot4dotmodulated_convdotmodulation(style).view(1, 1, 256, 1, 1)
        weight = self.stylegan_decoderdotto_rgbsdot4dotmodulated_convdotweight * style     
        weight = weight.view(3, 256, 1, 1)
        b, c, h, w = x.shape
        x = x.view(1, b * c, h, w)
        out = F.conv2d(x, weight, padding=0, groups=b)
        out = out.view(1, 3, 128, 128)
        out = out + self.stylegan_decoderdotto_rgbsdot4dotbias
        skip = F.interpolate(skip, scale_factor=2, mode='bilinear', align_corners=False)
        skip = out + skip

        # i = 11
        out = out0
        x = out
        b, c, h, w = 1, 256, 128, 128
        i += 2
        style = latent[:, i]   # 数值一致 
        style = self.stylegan_decoderdotstyle_convsdot10dotmodulated_convdotmodulation(latent[:, i]).view(b, 1, c, 1, 1)     
        #conv1
        weight = self.stylegan_decoderdotstyle_convsdot10dotmodulated_convdotweight * style    
        # self.demodulate = True:
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-08)
        weight = weight * demod.view(b, 128, 1, 1, 1)
        weight = weight.view(b * 128, 256, 3, 3) 
        # self.sample_mode == 'upsample'
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        b, c, h, w = x.shape
        out = F.conv2d(x, weight, padding=1, groups=b)
        out = out.view(1, 128, 256, 256) * 2 ** 0.5 
        b, _, h, w = 1, _, 256, 256
        noise = noise_dict[w]
        out = out + self.stylegan_decoderdotstyle_convsdot10dotweight * noise
        out = out + self.stylegan_decoderdotstyle_convsdot10dotbias
        out = self.activate(out)
        out_same, out_sft = torch.split(out, int(out.size(1) // 2), dim=1)
        out_sft = out_sft * conditions[i - 1] + conditions[i]
        out = torch.cat([out_same, out_sft], dim=1)
        #conv2
        style = self.stylegan_decoderdotstyle_convsdot11dotmodulated_convdotmodulation(latent[:, i + 1]).view(1, 1, 128, 1, 1)
        weight = self.stylegan_decoderdotstyle_convsdot11dotmodulated_convdotweight * style 
        # self.demodulate = True:
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-08)
        weight = weight * demod.view(1, 128, 1, 1, 1)
        weight = weight.view(b * 128, 128, 3, 3)
        out = F.conv2d(out, weight, padding=1, groups=b)
        out = out.view(1, 128, 256, 256) * 2 ** 0.5 
        noise = noise_dict[w]
        out = out + self.stylegan_decoderdotstyle_convsdot11dotweight * noise
        out = out + self.stylegan_decoderdotstyle_convsdot11dotbias
        out = self.activate(out)  
        out0 = out
        #to_rgb
        x = out
        style = latent[:, i + 2]
        style = self.stylegan_decoderdotto_rgbsdot5dotmodulated_convdotmodulation(style).view(1, 1, 128, 1, 1)
        weight = self.stylegan_decoderdotto_rgbsdot5dotmodulated_convdotweight * style     
        weight = weight.view(3, 128, 1, 1)
        b, c, h, w = x.shape
        x = x.view(1, b * c, h, w)
        out = F.conv2d(x, weight, padding=0, groups=b)
        out = out.view(1, 3, 256, 256) 
        out = out + self.stylegan_decoderdotto_rgbsdot5dotbias
        skip = F.interpolate(skip, scale_factor=2, mode='bilinear', align_corners=False)
        skip = out + skip
        
        # i = 13
        out = out0
        x = out
        b, c, h, w = 1, 128, 256, 256
        i += 2
        style = latent[:, i]   # 数值一致 
        style = self.stylegan_decoderdotstyle_convsdot12dotmodulated_convdotmodulation(latent[:, i]).view(b, 1, c, 1, 1)     
        #conv1
        weight = self.stylegan_decoderdotstyle_convsdot12dotmodulated_convdotweight * style    
        
        # self.demodulate = True:
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-08)
        weight = weight * demod.view(b, 64, 1, 1, 1)
        weight = weight.view(b * 64, 128, 3, 3) 
        
        # self.sample_mode == 'upsample'
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        b, c, h, w = x.shape
        out = F.conv2d(x, weight, padding=1, groups=b)
        out = out.view(1, 64, 512, 512) * 2 ** 0.5 
        b, _, h, w = 1, _, 512, 512
        noise = noise_dict[w]
        out = out + self.stylegan_decoderdotstyle_convsdot12dotweight * noise
        out = out + self.stylegan_decoderdotstyle_convsdot12dotbias
        out = self.activate(out)
        
        out_same, out_sft = torch.split(out, int(out.size(1) // 2), dim=1)
        out_sft = out_sft * conditions[i - 1] + conditions[i]
        out = torch.cat([out_same, out_sft], dim=1)
        #conv2
        style = self.stylegan_decoderdotstyle_convsdot13dotmodulated_convdotmodulation(latent[:, i + 1]).view(1, 1, 64, 1, 1)
        weight = self.stylegan_decoderdotstyle_convsdot13dotmodulated_convdotweight * style 
        
        # self.demodulate = True:
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-08)
        weight = weight * demod.view(1, 64, 1, 1, 1)
        weight = weight.view(b * 64, 64, 3, 3)
        out = F.conv2d(out, weight, padding=1, groups=b)
        out = out.view(1, 64, 512, 512) * 2 ** 0.5 
        noise = noise_dict[w]
        out = out + self.stylegan_decoderdotstyle_convsdot13dotweight * noise
        out = out + self.stylegan_decoderdotstyle_convsdot13dotbias
        out = self.activate(out)    
        out0 = out
        #to_rgb
        x = out
        style = latent[:, i + 2]
        style = self.stylegan_decoderdotto_rgbsdot6dotmodulated_convdotmodulation(style).view(1, 1, 64, 1, 1) 
        weight = self.stylegan_decoderdotto_rgbsdot6dotmodulated_convdotweight * style     
        weight = weight.view(3, 64, 1, 1)
        b, c, h, w = x.shape
        x = x.view(1, b * c, h, w)
        out = F.conv2d(x, weight, padding=0, groups=b)    
        out = out.view(1, 3, 512, 512) 
        out = out + self.stylegan_decoderdotto_rgbsdot6dotbias
        skip = F.interpolate(skip, scale_factor=2, mode='bilinear', align_corners=False)
        skip = out + skip
        return skip        

