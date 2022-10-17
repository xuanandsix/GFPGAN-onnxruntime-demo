# -*- coding: utf-8 -*-

#import cv2
import numpy as np
import time
import torch
import pdb
from collections import OrderedDict

import sys
sys.path.append('.')
sys.path.append('./lib')
import torch.nn as nn
from torch.autograd import Variable
import onnxruntime
import timeit

import argparse
from GFPGANReconsitution import GFPGAN

parser = argparse.ArgumentParser("ONNX converter")
parser.add_argument('--src_model_path', type=str, default=None, help='src model path')
parser.add_argument('--dst_model_path', type=str, default=None, help='dst model path')
parser.add_argument('--img_size', type=int, default=None, help='img size')
args = parser.parse_args()
    
#device = torch.device('cuda')
model_path = args.src_model_path
onnx_model_path = args.dst_model_path
img_size = args.img_size

model = GFPGAN()#.cuda()

x = torch.rand(1, 3, 512, 512)#.cuda()

state_dict = torch.load(model_path)['params_ema']
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    # stylegan_decoderdotto_rgbsdot1dotmodulated_convdotbias
    if "stylegan_decoder" in k:
        k = k.replace('.', 'dot')
        new_state_dict[k] = v
        k = k.replace('dotweight', '.weight')
        k = k.replace('dotbias', '.bias')
        new_state_dict[k] = v
    else:
        new_state_dict[k] = v
     
model.load_state_dict(new_state_dict, strict=False)
model.eval()

torch.onnx.export(model, x, onnx_model_path,
                    export_params=True, opset_version=11, do_constant_folding=True,
                    input_names = ['input'],output_names = [])


####
try:
    original_model = onnx.load(onnx_model_path)
    passes = ['fuse_bn_into_conv']
    optimized_model = optimizer.optimize(original_model, passes)
    onnx.save(optimized_model, onnx_model_path)
except:
    print('skip optimize.')

####
ort_session = onnxruntime.InferenceSession(onnx_model_path)
for var in ort_session.get_inputs():
    print(var.name)
for var in ort_session.get_outputs():
    print(var.name)
_,_,input_h,input_w = ort_session.get_inputs()[0].shape
t = timeit.default_timer()

img = np.zeros((input_h,input_w,3))

img = (np.transpose(np.float32(img[:,:,:,np.newaxis]), (3,2,0,1)) )#*self.scale

img = np.ascontiguousarray(img)
#    
ort_inputs = {ort_session.get_inputs()[0].name: img}
ort_outs = ort_session.run(None, ort_inputs)

print('onnxruntime infer time:', timeit.default_timer()-t)
print(ort_outs[0].shape)

# python torch2onnx.py  --src_model_path ./experiments/pretrained_models/GFPGANCleanv1-NoCE-C2.pth --dst_model_path ./GFPGAN.onnx --img_size 512 

# 新版本


# wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth

# python torch2onnx.py  --src_model_path ./GFPGANv1.4.pth --dst_model_path ./GFPGANv1.4.onnx --img_size 512 

# python torch2onnx.py  --src_model_path ./GFPGANCleanv1-NoCE-C2.pth --dst_model_path ./GFPGANv1.2.onnx --img_size 512 
