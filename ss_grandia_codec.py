# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 17:45:00 2020

@author: nanashi
"""

from decord import VideoReader
from decord import cpu
from PIL import Image
from matplotlib.pyplot import imshow
import numpy as np

dct_mat1 = np.array([
    [ 2.88223759,-2.51030053,-0.50810609, 1.86601668,-1.27954718,-3.1773366 , 2.78152966, 2.77393359],
    [ 0.45271213, 0.91632663, 0.08420463,-1.49153464, 1.49153464,-0.08420463,-0.91632663,-0.45271213],
    [-0.09837887,-0.79782624, 0.79782624, 0.09837887, 0.09837887, 0.79782624,-0.79782624,-0.09837887],
    [ 0.91632663, 1.49153464,-0.45271213,-0.08420463, 0.08420463, 0.45271213,-1.49153464,-0.91632663],
    [-0.85355339, 0.85355339, 0.85355339,-0.85355339,-0.85355339, 0.85355339, 0.85355339,-0.85355339],
    [ 0.08420463,-0.45271213,-1.49153464, 0.91632663,-0.91632663, 1.49153464, 0.45271213,-0.08420463],
    [-0.79782624, 0.09837887,-0.09837887, 0.79782624, 0.79782624,-0.09837887, 0.09837887,-0.79782624],
    [-1.49153464,-0.08420463, 0.91632663,-0.45271213, 0.45271213,-0.91632663, 0.08420463, 1.49153464]])

dct_mat2 = np.array([
    [0.35355339, 0.49039264, 0.46193977, 0.41573481, 0.35355339, 0.27778512, 0.19134172, 0.09754516],
    [0.35355339, 0.41573481, 0.19134172,-0.09754516,-0.35355339,-0.49039264,-0.46193977,-0.27778512],
    [0.35355339, 0.27778512,-0.19134172,-0.49039264,-0.35355339, 0.09754516, 0.46193977, 0.41573481],
    [0.35355339, 0.09754516,-0.46193977,-0.27778512, 0.35355339, 0.41573481,-0.19134172,-0.49039264],
    [0.35355339,-0.09754516,-0.46193977, 0.27778512, 0.35355339,-0.41573481,-0.19134172, 0.49039264],
    [0.35355339,-0.27778512,-0.19134172, 0.49039264,-0.35355339,-0.09754516, 0.46193977,-0.41573481],
    [0.35355339,-0.41573481, 0.19134172, 0.09754516,-0.35355339, 0.49039264,-0.46193977, 0.27778512],
    [0.35355339,-0.49039264, 0.46193977,-0.41573481, 0.35355339,-0.27778512, 0.19134172,-0.09754516]])

zigzag = np.array([
     0,  1,  8, 16,  9,  2,  3, 10, 
    17, 24, 32, 25, 18, 11,  4,  5, 
    12, 19, 26, 33, 40, 48, 41, 34, 
    27, 20, 13,  6,  7, 14, 21, 28, 
    35, 42, 49, 56, 57, 50, 43, 36, 
    29, 22, 15, 23, 30, 37, 44, 51, 
    58, 59, 52, 45, 38, 31, 39, 46, 
    53, 60, 61, 54, 47, 55, 62, 63], dtype=int)

def dct(block):
    return np.dot(np.dot(dct_mat1, block), dct_mat2)

def rgb2ycbcr_min(r, g, b):
    y = 0.25*r+0.5*g+0.25*b
    cb = (y - b)/2
    cr = (y - r)/2
    y -= 128
    
    return y, cb, cr

def subsample420_discard(chan):
    return chan[::2,::2]

def gen_dct_mat1():
    
    res = np.zeros((8,8))
    
    for j in range(0,8):
        for k in range(0,8):
            if j == 0:
                en = np.sqrt(2)/2
            else:
                en = 1
                
            res[j,k] = en*np.cos((j*(2*k+1)*np.pi)/16)
            if res[j,k] < 0:
                res[j,k] += 2
            
    return (0.5*res).T

def gen_blocks(chan, bsize=8):
    block_width = int(chan.shape[1]/bsize)
    block_height = int(chan.shape[0]/bsize)
    
    lines = np.split(chan, block_height)
    res = []
    for i in range(0, block_height):
        res.append(np.split(lines[i], block_width, axis=1))
                   
    return res

def process_block(block):
    block_dct = dct(block)
    block_quant = block_dct / quant[quant_level]
    return np.round(block_quant).astype(int)

quant = np.load("quantize.npy")
quant_level = 5

vr = VideoReader("examples/stock_video.mp4", ctx=cpu(0))

frame = Image.fromarray(vr[0].asnumpy(), mode='RGB')
frame = frame.resize((352, 198), resample=Image.LANCZOS)
frame = frame.crop((0, 11, 352, 187))
frame = np.array(frame)
#imshow(frame)

r_chan = frame[:,:,0].astype(float)
g_chan = frame[:,:,1].astype(float)
b_chan = frame[:,:,2].astype(float)

y, cb, cr = rgb2ycbcr_min(r_chan, g_chan, b_chan)
cb = subsample420_discard(cb)
cr = subsample420_discard(cr)

y_blocks = gen_blocks(y)
cb_blocks = gen_blocks(cb)
cr_blocks = gen_blocks(cr)

y_blocks = [[process_block(block) for block in line] for line in y_blocks]
cb_blocks = [[process_block(block) for block in line] for line in cb_blocks]
cr_blocks = [[process_block(block) for block in line] for line in cr_blocks]

