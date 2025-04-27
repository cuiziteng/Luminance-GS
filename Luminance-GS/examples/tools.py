import imghdr
from tkinter import Y
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Curve Mapping: mapping the input tensor with curve
def LUT_mapping(ts, lut_1d):
    t, t_min, t_max = ts[0], ts[1], ts[2]
    t = torch.clamp(t, 0, 1)
    H, W = t.shape[1], t.shape[2]
    range = lut_1d.shape[-1]-1
    t = (t*range).to(torch.int32)
    N_new = H * W
    id = t.to(torch.long).view(N_new)
    out = lut_1d.reshape(range+1)[id]
    out = out.reshape(H, W).unsqueeze(0)
    return out

def gamma_curve(x, g):
    # Power Curve, Gamma Correction
    y = torch.clamp(x, 1e-3, 1)
    y = y ** g
    return y

def s_curve(x, alpha, beta):
    below_alpha = x <= alpha
    epsilon = 1e-3
    s_below_alpha = alpha - alpha * ((1 - (x / alpha)) ** beta)
    s_above_alpha = alpha + (1 - alpha) * (((x - alpha) / (1 - alpha)) ** beta)
    return torch.where(below_alpha, s_below_alpha, s_above_alpha)


def value_encode(value, max, min):
    return (value-min)/(max-min)

def value_decode(value, max ,min):
    return value*(max-min)+min

def cal_min_max(normal, normal_sign, bias):
    normal_max = normal[:,0,:,:]*normal_sign[:,0,:,:] + normal[:,1,:,:]*normal_sign[:,1,:,:] + normal[:,2,:,:]*normal_sign[:,2,:,:] + bias
    normal_min = normal[:,0,:,:]*(1-normal_sign[:,0,:,:]) + normal[:,1,:,:]*(1-normal_sign[:,1,:,:]) + normal[:,2,:,:]*(1-normal_sign[:,2,:,:]) + bias
    return normal_max, normal_min

# img: [B, H, W, 3], normal:[B, 3] normal2:[B, 2]
def pixel_project(img, normal, normal2, bias):
    
    img = img.permute(0,2,3,1) # (B, H, W, C)
    R, G, B = img[:, :, :, 0], img[:, :, :, 1], img[:, :, :, 2]
    
    normal2 = torch.stack([-(normal2[:,0]*normal[:,1] + normal2[:,1]*normal[:,2])/normal[:,0], \
                            normal2[:,0], normal2[:,1]],dim=1)
    normal3 = torch.stack([normal[:,1]*normal2[:,2] - normal[:,2]*normal2[:,1], \
                        normal[:,2]*normal2[:,0] - normal[:,0]*normal2[:,2], \
                        normal[:,0]*normal2[:,1] - normal[:,1]*normal2[:,0]], dim=1)

    normal = F.normalize(normal,dim=1).unsqueeze(-1).unsqueeze(-1)
    normal2 = F.normalize(normal2,dim=1).unsqueeze(-1).unsqueeze(-1)
    normal3 = F.normalize(normal3,dim=1).unsqueeze(-1).unsqueeze(-1)
    
    t1 = R * normal[:,0,:,:] + G * normal[:,1,:,:] + B * normal[:,2,:,:] + bias[:, 0].unsqueeze(-1).unsqueeze(-1)
    t2 = R * normal2[:,0,:,:] + G * normal2[:,1,:,:] + B * normal2[:,2,:,:] + bias[:, 1].unsqueeze(-1).unsqueeze(-1) 
    t3 = R * normal3[:,0,:,:] + G * normal3[:,1,:,:] + B * normal3[:,2,:,:] + bias[:, 2].unsqueeze(-1).unsqueeze(-1) 

    normal_sign = torch.clip(torch.sign(normal), 0, 1)
    normal2_sign = torch.clip(torch.sign(normal2), 0, 1)
    normal3_sign = torch.clip(torch.sign(normal3), 0, 1)
    
    t1_max, t1_min = cal_min_max(normal, normal_sign, bias[:, 0].unsqueeze(-1).unsqueeze(-1))
    
    t2_max, t2_min = cal_min_max(normal2, normal2_sign, bias[:, 1].unsqueeze(-1).unsqueeze(-1))
    
    t3_max, t3_min = cal_min_max(normal3, normal3_sign, bias[:, 2].unsqueeze(-1).unsqueeze(-1))
    
    t1 = value_encode(t1,t1_max,t1_min)
    t2 = value_encode(t2,t2_max,t2_min)
    t3 = value_encode(t3,t3_max,t3_min)
    
    return [t1, normal, t1_max, t1_min], [t2, normal2, t2_max, t2_min], [t3, normal3, t3_max, t3_min], bias
    

# Project back images with learnable affine transformation & bias
def pixel_project_back(t1s, t2s, t3s, bias):
    t1 = value_decode(t1s[0], t1s[2], t1s[3]) - bias[:, 0].unsqueeze(-1).unsqueeze(-1)
    t2 = value_decode(t2s[0], t2s[2], t2s[3]) - bias[:, 1].unsqueeze(-1).unsqueeze(-1)
    t3 = value_decode(t3s[0], t3s[2], t3s[3]) - bias[:, 2].unsqueeze(-1).unsqueeze(-1)
    
    R_new = t1*t1s[1][:,0,:,:] + t2*t2s[1][:,0,:,:] + t3*t3s[1][:,0,:,:]
    G_new = t1*t1s[1][:,1,:,:] + t2*t2s[1][:,1,:,:] + t3*t3s[1][:,1,:,:]
    B_new = t1*t1s[1][:,2,:,:] + t2*t2s[1][:,2,:,:] + t3*t3s[1][:,2,:,:]
    
    img_out = torch.stack([R_new,G_new,B_new],dim=-1).permute(0,3,1,2)
    return img_out