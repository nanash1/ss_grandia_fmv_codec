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

def gen_blocks(chan, bsize=8):
    """
    Generates macro blocks from from channel data

    Parameters
    ----------
    chan : Array of float
        Channel data as numpy array.
    bsize : int, optional
        Block size. The default is 8.

    Returns
    -------
    res : list
        Nested list with blocks.

    """
    block_width = int(chan.shape[1]/bsize)
    block_height = int(chan.shape[0]/bsize)
    
    lines = np.split(chan, block_height)
    res = []
    for i in range(0, block_height):
        res.append(np.split(lines[i], block_width, axis=1))
    return res

def find_min(block, bsize=8):
    
    idx = bsize - 1
    while idx > -1:
        if np.any(block[:,idx]):
            break
        idx -= 1
    rows = idx
    
    idx = bsize - 1
    while idx > -1:
        if np.any(block[idx,:]):
            break
        idx -= 1
    cols = idx
    return (cols+1, rows+1)

def process_block(block):
    block = dct(block)
    block /= quant[quant_level]
    block = np.round(block).astype(int)
    dims = find_min(block)
    return block, dims

def gen_code(num_zeros, num, idx, codes, nums):
    abs_num = abs(num)
    
    full = 0
    part = 0
    if num_zeros > 0:
        full = int(num_zeros / 7)
        part = num_zeros % 7
        
    codes += [29]*full
    nums += [(7,3)]*full
    
    if abs_num == 1:
        if part == 5:
            codes += [28]
            part = 4
        elif part == 6:
            codes += [29]
            nums += [(6,3)]
            part = 0
            
        if num == -1:
            part += 5
        codes += [part]
        return
        
    elif abs_num == 2:
        if part > 3:
            codes += [29]
            nums += [(part,3)]
            part = 0
        elif part == 2:
            codes += [28]
            part = 1
            
        code = 10
        if num == -2:
            code += 1
        if part == 1:
            code += 2
        codes += [code]
        return
        
    elif part > 0 or full > 0:
        if part == 1:
            codes += [28]
        else:
            codes += [29]
            nums += [(part,3)]
            
    if abs_num < 5:
        code = 14
        if abs_num == 4:
            code += 2
        
    elif abs_num < 7:
        code = 18
        nums += [(abs_num-5, 1)]
        
    elif abs_num < 11:
        code = 20
        nums += [(abs_num-7, 2)]
        
    elif abs_num < 19:
        code = 22
        nums += [(abs_num-11, 3)]
        
    elif abs_num < 35:
        code = 24
        nums += [(abs_num-19, 4)]
    else:
        code = 26
        if idx == 0:
            nums += [(abs_num-35, 11)]
        elif idx < 3:
            nums += [(abs_num-35, 10)]
        else:
            nums += [(abs_num-35, 8)]
            
    if num < 0:
        code += 1 
    codes += [code]

def compress(block, dims, codes, nums, last_elem):
    length = dims[0]*dims[1]
    block = block.T.reshape(length)
    block = block[order[dims[0]-1, dims[1]-1]]
    zero_cntr = 0
    first_elem = block[0]
    block[0] -= last_elem
    for i in range(0,length-1):
        elem = block[i]
        if elem == 0:
            zero_cntr += 1
        else:
            gen_code(zero_cntr, elem, i, codes, nums)
            zero_cntr = 0
    codes += [30]
    return first_elem

order = np.load("order.npy", allow_pickle=True).reshape((8,8))
quant = np.load("quantize.npy")
quant_level = 17

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

codes = []
nums = []
diff = 0
for line in y_blocks:
    diff = 0
    for block in line:
        diff = compress(*block, codes, nums, diff)