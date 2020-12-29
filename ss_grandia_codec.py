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
import math

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

def reduce_block(block):
    '''
    Reduces the block size by finding the non-zero subblock of a block

    Parameters
    ----------
    block : numpy array
        Macroblock matrix.

    Returns
    -------
    numpy array
        Macroblock matrix.

    '''
    x, y = np.nonzero(block)
    if len(x) > 0:
        x = x.max()
    else:
        x = 0
    if len(y) > 0:
        y = y.max()
    else:
        y = 0
    return block[:x+1,:y+1]

def preprocess_block(block):
    '''
    Executes processing steps that are indepentent per block:
        - DCT
        - Quantization
        - Dimensional reduction

    Parameters
    ----------
    block : numpy array
        Macroblock matrix.

    Returns
    -------
    block : numpy array
        Processed block.

    '''
    block = dct(block)
    block /= quant[quant_level]
    block = np.round(block).astype(int)
    block = reduce_block(block)
    return block

def gen_code(num_zeros, elem, idx, codes, nums):
    '''
    Generates block compression codes

    Parameters
    ----------
    num_zeros : int
        Number of unwritten zeros.
    elem : int
        Element value.
    idx : int
        index of element in flattened macroblock.
    codes : list
        new codes are appended to this list.
    nums : list of (int, int)
        Additional data is appended to this list (number, size in bits).

    Returns
    -------
    None.

    '''
    abs_elem = abs(elem)
    
    full = 0
    part = 0
    if num_zeros > 0:
        full = int(num_zeros / 7)
        part = num_zeros % 7
        
    codes += [29]*full
    nums += [(7,3)]*full
    
    if abs_elem == 1:
        if part == 5:
            codes += [28]
            part = 4
        elif part == 6:
            codes += [29]
            nums += [(6,3)]
            part = 0
            
        if elem == -1:
            part += 5
        codes += [part]
        return
        
    elif abs_elem == 2:
        if part > 3:
            codes += [29]
            nums += [(part,3)]
            part = 0
        elif part == 2:
            codes += [28]
            part = 1
            
        code = 10
        if elem == -2:
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
            
    if abs_elem < 5:
        code = 14
        if abs_elem == 4:
            code += 2
        
    elif abs_elem < 7:
        code = 18
        nums += [(abs_elem-5, 1)]
        
    elif abs_elem < 11:
        code = 20
        nums += [(abs_elem-7, 2)]
        
    elif abs_elem < 19:
        code = 22
        nums += [(abs_elem-11, 3)]
        
    elif abs_elem < 35:
        code = 24
        nums += [(abs_elem-19, 4)]
    else:
        code = 26
        if idx == 0:
            nums += [(abs_elem-35, 11)]
        elif idx < 3:
            nums += [(abs_elem-35, 10)]
        else:
            nums += [(abs_elem-35, 8)]
            
    if elem < 0:
        code += 1 
    codes += [code]

def compress(block, codes, nums, dim_codes, last_elem):
    '''
    Generates compressed data from preproccessed blocks

    Parameters
    ----------
    block : numpy array
        preproccessed macro block.
    codes : list of int
        Compression codes are appended to this list.
    nums : list of (int, int)
        Additional data is appended to this list (number, size in bits).
    dim_codes : list of int
        Dimension codes are appended to this list.
    last_elem : int
        previous first element.

    Returns
    -------
    first_elem : int
        first element.

    '''
    cols = block.shape[0]
    rows = block.shape[1]
    length = cols*rows
    dim_codes += [(cols-1 << 3) | (rows-1)]
    block = block.T.reshape(length)
    block = block[order[cols-1, rows-1]]
    zero_cntr = 0
    first_elem = block[0]
    block[0] -= last_elem
    for i in range(0,length):
        elem = block[i]
        if elem == 0:
            zero_cntr += 1
        else:
            gen_code(zero_cntr, elem, i, codes, nums)
            zero_cntr = 0
    codes += [30]
    return first_elem

def gen_start_cond(n:int):
    '''
    Generates the starting weights that are ideal if all elements are equally
    distributed

    Parameters
    ----------
    n : int
        Number of elements.

    Returns
    -------
    np.array
        Array with the weight of each element.

    '''
    x = int(math.log(n)/math.log(2))
    i = n - 2**x
    
    return np.array((n-2*i)*[1/(2**x)]+(2*i)*[1/(2**(x+1))])

def find_weights(counts):
    '''
    Finds the ideal weights for each element so that the compression is minimal

    Parameters
    ----------
    counts : dict
        Dictonary that contains the elements as keys and their number of
        occurrence as value.

    Returns
    -------
    codes : numpy array
        Codes as numpy array.
    weights : numpy array
        Weights for each code as numpy array.

    '''
    codes = np.array(list(counts.keys()))
    counts = np.array(list(counts.values()))
    sort_idx = np.argsort(counts)[::-1]
    codes = codes[sort_idx]
    counts = counts[sort_idx]
    weights = gen_start_cond(len(codes))
    
    log2 = math.log(2)
    for i in range(0, len(counts)-1):
        pool = []
        cref = counts[i]
        wsum = 0
        csum = 0
        for j in range(len(counts)-1, i, -1):
            if weights[j] > 1/256:
                if csum + counts[j] > cref:
                    break
                else:
                    csum += counts[j]
                    wsum += weights[j]
                    pool.append(j)
        if wsum == 0:
            break
        mult = wsum / weights[i]
        mult = int(math.log(mult)/log2)
        if mult < 1:
            continue
        mult = 2**mult
        
        wref = weights[i]*mult
        wsum = 0
        cntr = 0
        for idx in pool:
            cntr += 1
            wsum += weights[idx]
            if wsum == wref:
                weights[i] *= mult
                weights[pool[:cntr]] /= mult
                break
    
    weights = np.log(1/weights)/log2
    return codes, weights

def gen_sec1(y0, y1, cb, cr):
    sec1 = bytearray(1452)
    bytepos = 0
    for i in range(0, len(cb)):
        sec1[bytepos] = y0[2*i] << 2
        sec1[bytepos] |= y0[2*i+1] >> 4
        sec1[bytepos+1] = (y0[2*i+1] << 4) & 0xff
        sec1[bytepos+1] |= y1[2*i] >> 2
        sec1[bytepos+2] = (y1[2*i] << 6) & 0xff
        sec1[bytepos+2] |= y1[2*i+1]
        sec1[bytepos+3] = cb[i] << 2
        sec1[bytepos+3] |= cr[i] >> 4
        sec1[bytepos+3] = (cr[i] << 4) & 0xff
        
        bytepos += 6 
    return sec1
    

def gen_sec2(code_lines):
    
    # analyse the generated codes to find best compression
    code_count = {}
    for line in code_lines:
        for code in line:
            if code not in code_count:
                code_count[code] = 1
            else:
                code_count[code] += 1
    codes, weights = find_weights(code_count)
    
    sec2_codes = bytearray(16)
    sec2 = bytearray(1048576)
    
    # generate binary code lut
    bin_lut = {}
    addr = 0
    lut_repeats = (0, 0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01)
    for idx, code in enumerate(codes):
        if code % 2:
            sec2_codes[int(code/2)] |= int(weights[idx])
        else:
            sec2_codes[int(code/2)] |= int(weights[idx]) << 4
            
        weight = int(weights[idx])
        bin_lut[code] = (addr >> (8-weight), weight)
        addr += lut_repeats[weight]
        
    bitpos= 0
    bytepos = 0
    line_pos = []
    for line in comp_codes:
        for code in line:
            shift = 8 - bitpos - bin_lut[code][1]
            bitpos += bin_lut[code][1]
            if shift < 0:
                sec2[bytepos] |= bin_lut[code][0] >> -shift
                bytepos += 1
                bitpos = -shift
                shift = 8 + shift
            sec2[bytepos] |= (bin_lut[code][0] << shift) & 0xff
            
        line_pos.append((bytepos, bitpos))
        
    return sec2_codes+sec2[:bytepos+1], line_pos
    
def gen_sec3(num_lines):
    sec3 = bytearray(1048576)
    bitpos= 8
    bytepos = 0
    line_pos = []
    for line in num_lines:
        for pair in line:
            bit_size = pair[1]
            while bit_size:
                shift = bit_size - bitpos
                if shift < 0:
                    sec3[bytepos] |= (pair[0] << -shift) & 0xff
                    bits_written = bit_size
                else:
                    sec3[bytepos] |= (pair[0] >> shift) & 0xff
                    bits_written = bitpos
                    bytepos += 1
                bitpos -= bits_written
                bit_size -= bits_written
                if bitpos == 0:
                    bitpos = 8
                    
        line_pos.append((bytepos, 8-bitpos))
    return sec3[:bytepos+1], line_pos

def gen_sec0(sec2_linepos, sec3_linepos):
    sec0 = bytearray(66)
    sec0[1] = 16
    
    bytepos = 6
    for i in range(0, len(sec2_linepos)-1):
        sec0[bytepos] = (sec2_linepos[i][0] + 15) >> 8                          # +15 because the 16 bytes of the decoding table are included
        sec0[bytepos+1] = (sec2_linepos[i][0] + 15) & 0xff
        sec0[bytepos+2] = (sec3_linepos[i][0] - 1) >> 8
        sec0[bytepos+3] = (sec3_linepos[i][0] - 1) & 0xff
        sec0[bytepos+4] = sec2_linepos[i][1]
        sec0[bytepos+5] = sec3_linepos[i][1]
        
        bytepos += 6
    return sec0

order = np.load("order.npy", allow_pickle=True).reshape((8,8))
quant = np.load("quantize.npy")
quant_level = 15

vr = VideoReader("examples/intro.mkv", ctx=cpu(0))

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

y_blocks = [[preprocess_block(block) for block in line] for line in y_blocks]
cb_blocks = [[preprocess_block(block) for block in line] for line in cb_blocks]
cr_blocks = [[preprocess_block(block) for block in line] for line in cr_blocks]

comp_codes = []
comp_nums = []

dim_codes_y0 = []
dim_codes_y1 = []
dim_codes_cb = []
dim_codes_cr = []

# generate compression codes and data for each line of the video frame
# store dimension codes seperately because they aren't stored by line
for i in range(0, len(cb_blocks)):
    comp_code_line = []
    comp_num_line = []
    diff = 0
    y_ind = int(i*2)
    for block in y_blocks[y_ind]:
        diff = compress(block, comp_code_line, comp_num_line, dim_codes_y0, diff)
    diff = 0
    for block in y_blocks[y_ind+1]:
        diff = compress(block, comp_code_line, comp_num_line, dim_codes_y1, diff)
    diff = 0
    for block in cb_blocks[i]:
        diff = compress(block, comp_code_line, comp_num_line, dim_codes_cb, diff)
    diff = 0
    for block in cr_blocks[i]:
        diff = compress(block, comp_code_line, comp_num_line, dim_codes_cr, diff)
    comp_codes.append(comp_code_line)
    comp_nums.append(comp_num_line)
    
# generate sections
sec1 = gen_sec1(dim_codes_y0, dim_codes_y1, dim_codes_cb, dim_codes_cr)
sec2, sec2_linepos = gen_sec2(comp_codes)
sec3, sec3_linepos = gen_sec3(comp_nums)
sec0 = gen_sec0(sec2_linepos, sec3_linepos)

# generate header
sec1_start = 10 + len(sec0)
sec2_start = sec1_start + len(sec1)
sec3_start = sec2_start + len(sec2)
if sec3_start % 4:
    padding = 4 - (sec3_start % 4)
    sec2 += bytearray(padding)
    sec3_start += padding
frame_header = bytearray(10)
frame_header[0] = 0b0 | quant_level
frame_header[1] = (quant_level << 4) | quant_level
frame_header[2] = 22
frame_header[3] = 11
frame_header[4] = sec1_start >> 8
frame_header[5] = sec1_start & 0xff
frame_header[6] = sec2_start >> 8
frame_header[7] = sec2_start & 0xff
frame_header[8] = sec3_start >> 8
frame_header[9] = sec3_start & 0xff

# write frame
written = 0
with open('test.bin', 'wb') as outest:
    written += outest.write(frame_header)
    written += outest.write(sec0)    
    written += outest.write(sec1)    
    written += outest.write(sec2)
    written += outest.write(sec3)
    if written % 0x800:
        padding = 0x800 - (written % 0x800)
        written += outest.write(bytearray(padding))
        