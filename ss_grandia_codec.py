# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 17:45:00 2020

@author: nanashi
"""
import numpy as np
import math

class codec:
    
    zigzag = np.array([
         0,  1,  8, 16,  9,  2,  3, 10, 
        17, 24, 32, 25, 18, 11,  4,  5, 
        12, 19, 26, 33, 40, 48, 41, 34, 
        27, 20, 13,  6,  7, 14, 21, 28, 
        35, 42, 49, 56, 57, 50, 43, 36, 
        29, 22, 15, 23, 30, 37, 44, 51, 
        58, 59, 52, 45, 38, 31, 39, 46, 
        53, 60, 61, 54, 47, 55, 62, 63], dtype=int)
    
    def __init__(self):
        self.colorspace = self.colorspace()
        self.lossy = self.lossy()
        self.lossless = self.lossless()
    
    class colorspace:

        @staticmethod
        def level_shift(channel: np.ndarray, vmin, vmax) -> np.ndarray:
            """
            Truncates values that are smaller than vmin and bigger than vmax

            Parameters
            ----------
            channel : np.ndarray
                Channel data.
            vmin : float
                Minimum channel value.
            vmax : float
                Maximum channel value.

            Returns
            -------
            np.ndarray
                Shifted array.

            """
            shape = channel.shape
            channel = channel.reshape(shape[0]*shape[1])
            idx = np.where(channel < vmin)
            channel[idx] = vmin
            idx = np.where(channel > vmax)
            channel[idx] = vmax
            return channel.reshape((shape[0], shape[1]))
        
        @staticmethod
        def scale(channel: np.ndarray, vmin, vmax) -> np.ndarray:
            
            channel *= (vmax-vmin)/255
            return channel + vmin

        @staticmethod
        def subsample420_discard(chan: np.ndarray) -> np.ndarray:
            """
            4:2:0 subsampling by discarding values 1,2,3 in a 4x4 field

            Parameters
            ----------
            chan : np.ndarray
                Channel data.

            Returns
            -------
            np.ndarray
                Subsampled channel.

            """
            return chan[::2,::2]
        
        @staticmethod
        def subsample420_avrg(chan: np.ndarray) -> np.ndarray:
            """
            4:2:0 subsampling by averaging values 0 and 2 in a 4x4 field

            Parameters
            ----------
            chan : np.ndarray
                Channel data.

            Returns
            -------
            np.ndarray
                Subsampled channel.

            """
            chan1 = chan[::2,::2]
            #chan2 = chan[::2,1::2]
            chan3 = chan[1::2,::2]
            #chan4 = chan[1::2,1::2]
            
            res = chan1+chan3
            return res/2
        
        def rgb2ycbcr_(self, r: np.ndarray, g: np.ndarray, b: np.ndarray):
            """
            Converts RGB colorspace to YCbCr without chroma subsampling

            Parameters
            ----------
            r : np.ndarray
                Red channel.
            g : np.ndarray
                Green channel.
            b : np.ndarray
                Blue channel.

            Returns
            -------
            y : np.ndarray
                Luminance channel.
            cb : np.ndarray
                DESCRIPTION.
            cr : np.ndarray
                DESCRIPTION.

            """
            #r = self.level_shift(r, 3, 251)
            #g = self.level_shift(g, 3, 251)
            #b = self.level_shift(b, 3, 251)
            y = 0.25*r+0.5*g+0.25*b
            cb = (y - b)/2
            cr = (y - r)/2
            self.scale(y, 16, 235)                                             # See Gibb's Phenomenon and https://en.wikipedia.org/wiki/YCbCr
            self.scale(cb, 16, 240)
            self.scale(cr, 16, 240)
            y -= 124
            return y, cb, cr
        
        def ycbcr2rgb_420(self, y0, y1, y2, y3, cb, cr):
            """
            Converts YCbCr colorspace to RGB with 4:2:0 subsampling

            Parameters
            ----------
            y0 : int
                Y upper left field.
            y1 : int
                Y upper right field.
            y2 : int
                Y lower left field.
            y3 : int
                Y lower right field.
            cb : int
                Cb field.
            cr : int
                Cr field.

            Returns
            -------
            rgb0 : tuple
                Upper left field.
            rgb1 : tuple
                Upper right field.
            rgb2 : tuple
                Lower left field.
            rgb3 : tuple
                Lower right field.

            """
            y0 += 124
            y1 += 124
            y2 += 124
            y3 += 124
            
            green0 = (y0 + cb + cr) & 0xff
            green1 = (y1 + cb + cr) & 0xff
            green2 = (y2 + cb + cr) & 0xff
            green3 = (y3 + cb + cr) & 0xff
            red0 = (-2*cr + y0) & 0xff
            red1 = (-2*cr + y2) & 0xff
            blue0 = (-2*cb + y0) & 0xff
            blue1 = (-2*cb + y2) & 0xff
                
            rgb0 = (red0, green0, blue0)
            rgb1 = (red0, green1, blue0)
            rgb2 = (red1, green2, blue1)
            rgb3 = (red1, green3, blue1)
    
            return rgb0, rgb1, rgb2, rgb3
        
        def ycbcr2rgb_test(self, y0, y1, y2, y3, cb, cr):
            """
            Converts YCbCr colorspace to RGB with 4:2:0 subsampling without adhering to
            the games overflow behavior

            Parameters
            ----------
            y0 : int
                Y upper left field.
            y1 : int
                Y upper right field.
            y2 : int
                Y lower left field.
            y3 : int
                Y lower right field.
            cb : int
                Cb field.
            cr : int
                Cr field.

            Returns
            -------
            rgb0 : tuple
                Upper left field.
            rgb1 : tuple
                Upper right field.
            rgb2 : tuple
                Lower left field.
            rgb3 : tuple
                Lower right field.

            """
            y0 += 124
            y1 += 124
            y2 += 124
            y3 += 124
            
            green0 = y0 + cb + cr
            green1 = y1 + cb + cr
            green2 = y2 + cb + cr
            green3 = y3 + cb + cr
            red0 = -2*cr + y0
            red1 = -2*cr + y2
            blue0 = -2*cb + y0
            blue1 = -2*cb + y2
                
            rgb0 = (red0, green0, blue0)
            rgb1 = (red0, green1, blue0)
            rgb2 = (red1, green2, blue1)
            rgb3 = (red1, green3, blue1)
    
            return rgb0, rgb1, rgb2, rgb3
        
        def rgb2ycbcr(self, r, g, b):
            y, cb, cr = self.rgb2ycbcr_(r, g, b)
            cb = self.subsample420_discard(cb)
            cr = self.subsample420_discard(cr)
            return y, cb, cr
        
    class block:
        
        @staticmethod
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
            blocks = []
            for i in range(0, block_height):
                blocks.append(np.split(lines[i], block_width, axis=1))
            return blocks
        
    class lossy:
        
        def __init__(self):
            self.quant = np.load("quantize.npy")
            self.dct = self.dct()
        
        class dct:
            
            dct_mat_1_fix = np.array(
                [0x5a82, 0x7d8a, 0x7642, 0x6a6e, 0x5a82, 0x471d, 0x30fc, 0x18f9,
                 0x5a82, 0x6a6e, 0x30fc,-0x18f9,-0x5a82,-0x7d8a,-0x7642,-0x471d,         
                 0x5a82, 0x471d,-0x30fc,-0x7d8a,-0x5a82, 0x18f9, 0x7642, 0x6a6e, 
                 0x5a82, 0x18f9,-0x7642,-0x471d, 0x5a82, 0x6a6e,-0x30fc,-0x7d8a, 
                 0x5a82,-0x18f9,-0x7642, 0x471d, 0x5a82,-0x6a6e,-0x30fc, 0x7d8a, 
                 0x5a82,-0x471d,-0x30fc, 0x7d8a,-0x5a82,-0x18f9, 0x7642,-0x6a6e, 
                 0x5a82,-0x6a6e, 0x30fc, 0x18f9,-0x5a82, 0x7d8a,-0x7642, 0x471d, 
                 0x5a82,-0x7d8a, 0x7642,-0x6a6e, 0x5a82,-0x471d, 0x30fc,-0x18f9
                 ], dtype="int64").reshape((8,8))
            
            dct_mat_21_fix = np.array(
                [[0x5a82, 0x5a82, 0x5a82, 0x5a82, 0x5a82, 0x5a82, 0x5a82, 0x5a82]], dtype='int64')
            dct_mat_22_fix = np.array(
                [[0x5a8200, 0x5a8200, 0x5a8200, 0x5a8200, 0x5a8200, 0x5a8200, 0x5a8200, 0x5a8200],
                 [0x7d8a80, 0x6a6080, 0x471d00, 0x192200,-0x192200,-0x471d00,-0x6a6080,-0x7d8a80]], dtype='int64')
            dct_mat_23_fix = np.array(
                [[0x5a8200, 0x5a8200, 0x5a8200, 0x5a8200, 0x5a8200, 0x5a8200, 0x5a8200, 0x5a8200],
                 [0x6a6d80, 0x7d8a80, 0x192200, 0x471d00,-0x471d00,-0x192200,-0x7d8a80,-0x6a6d80],
                 [0x30fc00, 0x764200,-0x764200,-0x30fc00,-0x30fc00,-0x764200, 0x764200, 0x30fc00]], dtype='int64')
            dct_mat_24_fix = np.array(
                [[ 0x5a8200, 0x5a8200, 0x5a8200, 0x5a8200, 0x5a8200, 0x5a8200, 0x5a8200, 0x5a8200],
                 [ 0x6a6d80, 0x7d8a80, 0x192200, 0x471d00,-0x471d00,-0x192200,-0x7d8a80,-0x6a6d80],
                 [ 0x30fc00, 0x764200,-0x764200,-0x30fc00,-0x30fc00,-0x764200, 0x764200, 0x30fc00],
                 [-0x18f900, 0x6a6e00,-0x471d00, 0x7d8a00,-0x7d8a00, 0x471d00,-0x6a6e00,-0x18f900]], dtype='int64')
            
            def __init__(self):
                self.dct_mat1 = self._gen_dct_mat()
                self.dct_mat2 = self._gen_dct_mat().T
                
            @staticmethod
            def _gen_dct_mat():
                res = np.zeros((8,8))
                for j in range(0,8):
                    for k in range(0,8):
                        if j == 0:
                            en = np.sqrt(2)/2
                        else:
                            en = 1
                        res[j,k] = en*np.cos((j*(2*k+1)*np.pi)/16)
                return 0.5*res
            
            def _idct_21(self, block):
                
                res = np.dot(block,self.dct_mat_21_fix)
                return np.right_shift(res, 32)
            
            def _idct_22(self, block):
                
                res1 = np.dot(block[:,0:1],self.dct_mat_22_fixblock[0:1,:])
                res2 = np.dot(block[:,1:2],self.dct_mat_22_fixblock[1:2,:])
                res = np.right_shift(res1, 32) + np.right_shift(res2, 32)
                res = np.right_shift(res1, 8)
                return res
            
            def _idct_23(self, block):
                
                res1 = np.dot(block[:,0:1],self.dct_mat_23_fixblock[0:1,:])
                res2 = np.dot(block[:,1:2],self.dct_mat_23_fixblock[1:2,:])
                res3 = np.dot(block[:,2:3],self.dct_mat_23_fixblock[2:3,:])
                res = np.right_shift(res1, 32) + np.right_shift(res2, 32) + np.right_shift(res3, 32)
                res = np.right_shift(res1, 8)
                return res
            
            def _idct_24(self, block):
                
                res1 = np.dot(block[:,0:1],self.dct_mat_24_fixblock[0:1,:])
                res2 = np.dot(block[:,1:2],self.dct_mat_24_fixblock[1:2,:])
                res3 = np.dot(block[:,2:3],self.dct_mat_24_fixblock[2:3,:])
                res4 = np.dot(block[:,3:4],self.dct_mat_24_fixblock[3:4,:])
                res = np.right_shift(res1, 32) + np.right_shift(res2, 32) + \
                    np.right_shift(res3, 32) + np.right_shift(res4, 32)
                res = np.right_shift(res1, 8)
                return res
            
            def _idct_25(self, block):
                
                res = np.empty((8,8), dtype="int64")
                for i in range(0,8):
                    fp = 0x5a8200*(block[i,0] + block[i,4]) >> 32
                    fy = 0x5a8200*(block[i,0] - block[i,4]) >> 32
                    
                    wtf = fy +((0x471d00*block[i,1])>>32) -((0x30fc00*block[i,2])>>32)                  # The fuck is this?!
                    wtf = (0x7d8a00*wtf)>>32
                    
                    res[i,0] = fy +((0x6a6d80*block[i,1])>>32) +((0x30fc00*block[i,2])>>32) -((0x18F900*block[i,3])>>32)
                    res[i,1] = fp +((0x7d8a00*block[i,1])>>32) +((0x764200*block[i,2])>>32) +((0x6a6d80*block[i,3])>>32)
                    res[i,2] = fp +((0x192200*block[i,1])>>32) -((0x764200*block[i,2])>>32) -((0x471d00*block[i,3])>>32)
                    res[i,3] = fy -((0x30fc00*block[i,1])>>32) +wtf
                    res[i,4] = fy -((0x30fc00*block[i,1])>>32) -wtf
                    res[i,5] = fp -((0x192200*block[i,1])>>32) -((0x764200*block[i,2])>>32) +((0x471d00*block[i,3])>>32)
                    res[i,6] = fp -((0x7d8a00*block[i,1])>>32) +((0x764200*block[i,2])>>32) -((0x6a6d80*block[i,3])>>32)
                    res[i,7] = fy -((0x6a6d80*block[i,1])>>32) +((0x30fc00*block[i,2])>>32) +((0x18F900*block[i,3])>>32)
                    
                return np.right_shift(res, 8)
            
            def _idct_28(self, block):
                
                res = np.empty((8,8), dtype="int64")
                for i in range(0,8):
                    fp = 0x5a8200*(block[i,0] + block[i,4]) >> 32
                    fy = 0x5a8200*(block[i,0] - block[i,4]) >> 32
                    fb = 0x764200*block[i,2] >> 32
                    fb += 0x30fc00*block[i,6] >> 32
                    fgo = (0x30fc00*block[i,2]) >> 32
                    fgo -= (0x764200*block[i,6]) >> 32
                    fgr = 0xb50400*(block[i,3] + block[i,5]) >> 32
                    fo = 0xb50400*(block[i,3] - block[i,5]) >> 32
                    
                    scal1 = block[i,1] >> 16
                    scal1 <<= 8
                    scal7 = block[i,7] >> 16
                    scal7 <<= 8
                    
                    fgrp = scal1 + fgr
                    fgrm = scal1 - fgr
                    fop = scal7 + fo
                    fom = scal7 - fo
                    
                    res[i,0] = fp +fb  +((0x7d8a8000*fgrp)>>32) +((0x19220000*fop)>>32)
                    res[i,1] = fy +fgo +((0x6a6d8000*fgrm)>>32) -((0x471d0000*fom)>>32)
                    res[i,2] = fy -fgo +((0x471d0000*fgrm)>>32) +((0x6a6d8000*fom)>>32)
                    res[i,3] = fp -fb  +((0x19220000*fgrp)>>32) -((0x7d8a8000*fop)>>32)
                    res[i,4] = fp -fb  -((0x19220000*fgrp)>>32) +((0x7d8a8000*fop)>>32)
                    res[i,5] = fy -fgo -((0x471d0000*fgrm)>>32) -((0x6a6d8000*fom)>>32)
                    res[i,6] = fy +fgo -((0x6a6d8000*fgrm)>>32) +((0x471d0000*fom)>>32)
                    res[i,7] = fp +fb  -((0x7d8a8000*fgrp)>>32) -((0x19220000*fop)>>32)
                    
                return np.right_shift(res, 8)
            
            def dct_f(self, block):
                return np.dot(np.dot(self.dct_mat1, block), self.dct_mat2)
            
            def idct_f(self, block):
                return np.dot(np.dot(self.dct_mat2, block), self.dct_mat1)
            
            def idct(self, block):
                
                rows = block.shape[0]
                step_1_res = np.dot(self.dct_mat_1_fix[:,0:rows], block)
                
                cols = step_1_res.shape[1]
                
                if cols == 1:
                    return self._idct_21(step_1_res)
                elif cols == 2:
                    return self._idct_22(step_1_res)
                elif cols == 3:
                    return self._idct_23(step_1_res)
                elif cols == 4:
                    return self._idct_24(step_1_res)
                elif cols == 5:
                    return self._idct_25(step_1_res)
                elif cols == 8:
                    return self._idct_28(step_1_res)
                else:
                    temp = np.zeros((8,8), dtype='int64')
                    temp[:,:cols] = step_1_res
                    return self._idct_28(temp)
            
        def encode(self, block, level):
            '''
            Executes lossy block encoding:
                - DCT
                - Quantization
        
            Parameters
            ----------
            block : numpy array
                Macroblock matrix.
        
            Returns
            -------
            block : numpy array
                Processed block.
        
            '''
            block = self.dct.dct_f(block)
            block /= self.quant[level]
            block = np.round(block).astype(int)
            return block
        
        def decode(self, block, level):
            
            block = block*self.quant[level][:block.shape[0],:block.shape[1]].astype('int32')
            return self.dct.idct(block)
        
    class lossless:
        
        def __init__(self):
            self.order = np.load("order.npy", allow_pickle=True).reshape((8,8))
        
        @staticmethod
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

        @staticmethod
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
                full = int(num_zeros / 9)
                part = num_zeros % 9
                
            codes += [29]*full
            nums += [(7,3)]*full                                                # offset by 2, so 7 := 9 zeros
            
            if abs_elem == 1:
                if part == 5:
                    codes += [28]
                    part = 4
                elif part > 5:
                    codes += [29]
                    nums += [(part-2,3)]
                    part = 0
                    
                if elem == -1:
                    part += 5
                codes += [part]
                return
                
            elif abs_elem == 2:
                if part > 3:
                    codes += [29]
                    nums += [(part-2,3)]
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
                
            elif part > 0:
                if part == 1:
                    codes += [28]
                else:
                    codes += [29]
                    nums += [(part-2,3)]
                    
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

        def gen_comp_data(self, block, codes, nums, dim_codes, last_elem):
            '''
            Generates compressed data from blocks
        
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
            block = self.reduce_block(block)
            rows = block.shape[0]
            cols = block.shape[1]
            length = cols*rows
            dim_codes += [(cols-1 << 3) | (rows-1)]
            block = block.T.reshape(length)
            block = block[self.order[cols-1, rows-1]]
            zero_cntr = 0
            first_elem = block[0]
            block[0] -= last_elem
            for i in range(0,length):
                elem = block[i]
                if elem == 0:
                    zero_cntr += 1
                else:
                    self.gen_code(zero_cntr, elem, i, codes, nums)
                    zero_cntr = 0
            codes += [30]
            return first_elem
        
        @staticmethod
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
        
        def find_weights(self, counts):
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
            weights = self.gen_start_cond(len(codes))
            
            log2 = math.log(2)
            for i in range(0, len(counts)-1):
                pool = []
                cref = counts[i]
                wsum = 0
                csum = 0
                min_weight = 1
                for j in range(len(counts)-1, i, -1):
                    if weights[j] > 1/256:
                        if csum + counts[j] > cref:
                            break
                        else:
                            csum += counts[j]
                            wsum += weights[j]
                            pool.append(j)
                            if weights[j] < min_weight:
                                min_weight = weights[j]
                if wsum == 0:
                    break
                mult = wsum / weights[i]
                mult = int(math.log(mult)/log2)
                if mult < 1:
                    continue
                mult = 2**mult
                while (min_weight / mult) < (1/256):
                    mult /= 2
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
        
        @staticmethod
        def gen_sec0(sec2_linepos, sec3_linepos):
            sec0 = bytearray(66)
            sec0[1] = 16
            
            bytepos = 6
            for i in range(0, len(sec2_linepos)-1):
                sec0[bytepos] = (sec2_linepos[i][0] + 16) >> 8                  # +16 because the 16 bytes of the decoding table are included
                sec0[bytepos+1] = (sec2_linepos[i][0] + 16) & 0xff
                sec0[bytepos+2] = (sec3_linepos[i][0]) >> 8
                sec0[bytepos+3] = (sec3_linepos[i][0]) & 0xff
                sec0[bytepos+4] = sec2_linepos[i][1]
                sec0[bytepos+5] = sec3_linepos[i][1]
                
                bytepos += 6
            return sec0
        
        @staticmethod
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
                sec1[bytepos+3] = cr[i] << 2
                sec1[bytepos+3] |= cb[i] >> 4
                sec1[bytepos+4] = (cb[i] << 4) & 0xff
                
                bytepos += 6 
            return sec1
        
        def gen_sec2(self, code_lines):
            
            # analyse the generated codes to find best compression
            code_count = {}
            for line in code_lines:
                for code in line:
                    if code not in code_count:
                        code_count[code] = 1
                    else:
                        code_count[code] += 1
            codes, weights = self.find_weights(code_count)
            
            sec2_codes = bytearray(16)
            sec2 = bytearray(1048576)
            
            # generate binary code lut
            sort_idx = np.argsort(codes)
            codes = codes[sort_idx]
            weights = weights[sort_idx]
            bin_lut = {}
            addr = 0
            lut_repeats = (0, 0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01)
            while True:
                idx = np.argmax(weights)
                max_code = int(weights[idx])
                if max_code == 0:
                    break
                code = codes[idx]
                if code % 2:
                    sec2_codes[int(code/2)] |= int(weights[idx])
                else:
                    sec2_codes[int(code/2)] |= int(weights[idx]) << 4
                bin_lut[code] = (addr >> (8-max_code), max_code)
                addr += lut_repeats[max_code]
                weights[idx] = 0
                
            bitpos= 0
            bytepos = 0
            line_pos = []
            for line in code_lines:
                for code in line:
                    shift = 8 - bitpos - bin_lut[code][1]
                    bitpos += bin_lut[code][1]
                    if shift < 0:
                        sec2[bytepos] |= bin_lut[code][0] >> -shift
                        bytepos += 1
                        bitpos = -shift
                        shift = 8 + shift
                    sec2[bytepos] |= (bin_lut[code][0] << shift) & 0xff
                    
                if bitpos == 8:
                    line_pos.append((bytepos+1, 0))
                else:
                    line_pos.append((bytepos, bitpos))
                
            return sec2_codes+sec2[:bytepos+1], line_pos
            
        @staticmethod
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
        
        def encode(self, y, cb, cr, level):
            
            comp_codes = []
            comp_nums = []
            
            dim_codes_y0 = []
            dim_codes_y1 = []
            dim_codes_cb = []
            dim_codes_cr = []
            
            # generate compression codes and data for each line of the video frame
            # store dimension codes seperately because they aren't stored by line
            for i in range(0, len(cb)):
                comp_code_line = []
                comp_num_line = []
                diff = 0
                y_ind = int(i*2)
                for block in y[y_ind]:
                    diff = self.gen_comp_data(block, comp_code_line, comp_num_line, dim_codes_y0, diff)
                diff = 0
                for block in y[y_ind+1]:
                    diff = self.gen_comp_data(block, comp_code_line, comp_num_line, dim_codes_y1, diff)
                diff = 0
                for block in cr[i]:
                    diff = self.gen_comp_data(block, comp_code_line, comp_num_line, dim_codes_cr, diff)
                diff = 0
                for block in cb[i]:
                    diff = self.gen_comp_data(block, comp_code_line, comp_num_line, dim_codes_cb, diff)
                comp_codes.append(comp_code_line)
                comp_nums.append(comp_num_line)
                
            # generate sections
            sec1 = self.gen_sec1(dim_codes_y0, dim_codes_y1, dim_codes_cb, dim_codes_cr)
            sec2, sec2_linepos = self.gen_sec2(comp_codes)
            sec3, sec3_linepos = self.gen_sec3(comp_nums)
            sec0 = self.gen_sec0(sec2_linepos, sec3_linepos)
            
            # generate header
            sec1_start = 10 + len(sec0)
            sec2_start = sec1_start + len(sec1)
            sec3_start = sec2_start + len(sec2)
            if sec3_start % 4:
                padding = 4 - (sec3_start % 4)
                sec2 += bytearray(padding)
                sec3_start += padding
            frame_header = bytearray(10)
            frame_header[0] = 0b0 | level
            frame_header[1] = (level << 4) | level
            frame_header[2] = 22
            frame_header[3] = 11
            frame_header[4] = sec1_start >> 8
            frame_header[5] = sec1_start & 0xff
            frame_header[6] = sec2_start >> 8
            frame_header[7] = sec2_start & 0xff
            frame_header[8] = sec3_start >> 8
            frame_header[9] = sec3_start & 0xff
            
            return frame_header+sec0+sec1+sec2+sec3