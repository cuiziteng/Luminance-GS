from turtle import distance
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from PIL import Image
eta=1e-8

# Curve Mapping: mapping the input tensor with curve
def LUT_mapping(ts, lut_1d):
    t, t_min, t_max = ts[0], ts[1], ts[2]
    H, W = t.shape[0], t.shape[1]
    range = lut_1d.shape[-1]-1
    t = (t*range).to(torch.int32)
    N_new = H * W
    id = t.to(torch.long).view(N_new)
    out = lut_1d.reshape(range+1)[id]
    out = out.reshape(H, W)
    return out

# Encode the Value with Max-Min Value
def value_encode(value, max, min):
    return (value-min)/(max-min+eta)

def value_decode(value, max ,min):
    return value*(max-min)+min

def gen_range(normal):
    # Corner Points on the RGB Cube
    
    points = [[1,0,0], [0,1,0], [0,0,1]]
    
    d = - 0.5* (normal[0]+ normal[1]+ normal[2])
    #d = 0
    distance_t = normal[0]**2 + normal[1]**2 + normal[2]**2
    ranges = []
    for point in points:
        t = -(normal[0]*point[0] + normal[1]*point[1] + normal[2]*point[2] + d)/distance_t
        point_flat = torch.stack([normal[0] * t + point[0], normal[1] * t + point[1], normal[2] * t + point[2]], dim=0)
        range = ((point_flat[0]-0.5)**2 + (point_flat[1]-0.5)**2 + (point_flat[2]-0.5)**2)**0.5
        ranges.append(range)
    
    t_min = -(normal[0] + normal[1] + normal[2] + d)/distance_t
    t_max = (normal[0] + normal[1] + normal[2] - d)/distance_t  
    radius_max = torch.max(torch.stack([i for i in ranges], dim=0))
    return t_min, t_max, radius_max

# Before Look Up Process
def pixel_project_2d(img, normal, normal2, t_min, t_max, distance_max):
    '''
    (1). Project the 3D pixel on 2D plane (Ax+By+Cz+D=0)
       Input: input image (img) ; learned normal of 2D plane (normal)
       Return: t in Parametric equation (t, t_max, t_min); projected image (img_2D)
    '''    
    img = img.permute(1,2,0)
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    d = -0.5* (normal[0]+ normal[1]+ normal[2])
    #d = 0
    distance = normal[0]**2 + normal[1]**2 + normal[2]**2
    #print(distance)
    t = -(normal[0]*R + normal[1]*G + normal[2]*B + d)/distance
    #print('the t is', t)
    R_new = normal[0]*t + R 
    #print(R_new)
    G_new = normal[1]*t + G
    #print(G_new)
    B_new = normal[2]*t + B
    # encode t in the range of 0~1 for tone curve mapping
    t = value_encode(t, t_max, t_min)
    # print('the t is:', t)
    # img_2D = torch.stack([R_new, G_new, B_new], dim=-1)
    ts = [t, t_max, t_min]
    
    '''
    (2). Translate 2D plane into polar coordinate system
       Input: projected image (img_2D) ; learned intial phase on 2D plane (normal2)
       Return: radius value (distance) ; angle 
    '''
    
    vector = torch.stack([R_new-0.5, G_new-0.5], dim=-1)
    
    vector = torch.stack([vector[:,:,0], vector[:,:,1], (-vector[:,:,0]*normal[0] - vector[:,:,1]*normal[1])/(normal[2]+1e-8)],dim=-1)
    
    distance = (vector[:,:,0]**2 + vector[:,:,1]**2 + vector[:,:,2]**2 + 1e-8)**0.5
    # print('the distance is', distance)
    # intial vector in the project coordinate
    normal2 = torch.stack([normal2[0], normal2[1], -(normal2[0]*normal[0] + normal2[1]*normal[1])/normal[2]])
    normal3 = torch.stack([normal[1]*normal2[2] - normal[2]*normal2[1], normal[2]*normal2[0] - normal[0]*normal2[2], \
                        normal[0]*normal2[1] - normal[1]*normal2[0]])
    normal2 = F.normalize(normal2, dim=0)
    normal3 = F.normalize(normal3, dim=0)
    normals = [normal, normal2, normal3]
    
    phase = torch.stack([(vector[:,:,1]*normal2[2] - vector[:,:,2]*normal2[1]), (vector[:,:,2]*normal2[0] - vector[:,:,0]*normal2[2]),\
                         (vector[:,:,0]*normal2[1] - vector[:,:,1]*normal2[0])], dim=-1)
    
    phase = - torch.sign(phase[:, :, 0]*normal[0] + phase[:, :, 1]*normal[1] + phase[:, :, 2]*normal[2])
    
    cos_theta = (vector[:,:,0]*normal2[0] + vector[:,:,1]*normal2[1] + vector[:,:,2]*normal2[2]) \
                /((distance * (normal2[0]**2+normal2[1]**2+normal2[2]**2)**0.5) + 1e-8)
    
    sin_theta = ((1 - cos_theta**2)**2)**0.25 * phase
    
    
    distance = value_encode(distance, distance_max, 0)
    
    distances = [distance, distance_max, 0]

    # return img_2D, t, t_max, t_min, distance, cos_theta, sin_theta
    return ts, normals, distances, cos_theta, sin_theta 

def pixel_project_back(normals, ts, distances, cos_theta, sin_theta):
    normal, normal2, normal3 = normals[0], normals[1], normals[2]
    t, t_max, t_min = ts[0], ts[1], ts[2]
    distance, distance_max = distances[0], distances[1]

    distance = value_decode(distance, distance_max, 0)
    # return to 2D plane
    img_2D = 0.5 + (distance*cos_theta).unsqueeze(-1)*normal2 + (distance*sin_theta).unsqueeze(-1)*normal3
    
    # decode t in the original range
    t = value_decode(t, t_max, t_min)
    
    R_new, G_new, B_new = img_2D[:, :, 0], img_2D[:, :, 1], img_2D[:, :, 2]
    R_re, G_re, B_re = R_new-normal[0]*t, G_new-normal[1]*t, B_new-normal[2]*t
    img_out = torch.stack([R_re, G_re, B_re], dim=-1)
    
    return img_out.permute(2,0,1)
    
if __name__ == '__main__':
    #rgb_img = torch.rand(1, 3, 256, 256)
    img = Image.open(r'/home/mil/cui/gsplat/data/LOM/buu/low/1.JPG')
    img = (np.asarray(img)/ 255.0)
    input = torch.from_numpy(img).float()
    input = input.permute(2,0,1)


    #img = torch.rand([3,200,300])
    normal = torch.Tensor([0.9, 0.9, 0.9])
    current_sum = normal.sum()
    normal = normal * (3.0 / current_sum)
    normal2 = torch.Tensor([1.5, 1.6])
    
    t_min, t_max, radius_max = gen_range(normal)
    print('t_min', t_min)
    print('t_max', t_max)
    print(radius_max)
    ts, normals, distances, cos_theta, sin_theta = pixel_project_2d(input, normal, normal2, t_min, t_max, radius_max)
    
    LUT_l = torch.nn.Parameter(torch.linspace(0,1,255), requires_grad=False).unsqueeze(0)
    LUT_r = torch.nn.Parameter(torch.linspace(0,1,100), requires_grad=False).unsqueeze(0)


    ts_out = [LUT_mapping(ts,LUT_l), ts[1], ts[2]]
    
    distances_out = [LUT_mapping(distances, LUT_r), distances[1], distances[2]]
    img_out = pixel_project_back(normals, ts_out, distances_out, cos_theta, sin_theta)
    
    
    

    
    