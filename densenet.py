#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:20:33 2019

@author: Shusil Dangi

References:
    https://github.com/ShusilDangi/DenseUNet-K
It is a simplied version of DenseNet with U-NET architecture.
2D implementation
"""

"""
For timing measurement, Ying added a few lines of code in this file
"""
import time

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class DenseNet2D_down_block(nn.Module):
    def __init__(self,input_channels,output_channels,down_size,dropout=False,prob=0):
        super(DenseNet2D_down_block, self).__init__()
        self.conv1 = nn.Conv2d(input_channels,output_channels,kernel_size=(3,3),padding=(1,1))
        self.conv21 = nn.Conv2d(input_channels+output_channels,output_channels,kernel_size=(1,1),padding=(0,0))
        self.conv22 = nn.Conv2d(output_channels,output_channels,kernel_size=(3,3),padding=(1,1))
        self.conv31 = nn.Conv2d(input_channels+2*output_channels,output_channels,kernel_size=(1,1),padding=(0,0))
        self.conv32 = nn.Conv2d(output_channels,output_channels,kernel_size=(3,3),padding=(1,1))
        self.max_pool = nn.AvgPool2d(kernel_size=down_size)            
        
        self.relu = nn.LeakyReLU()
        self.down_size = down_size
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)
        self.dropout3 = nn.Dropout(p=prob)
        self.bn = torch.nn.BatchNorm2d(num_features=output_channels)
    
    def forward(self, x):
        #time0 = time.time_ns()
        if self.down_size != None:
            x = self.max_pool(x)
        #time1 = time.time_ns()
        if self.dropout:
            x1 = self.relu(self.dropout1(self.conv1(x)))
            #time2 = time.time_ns()
            x21 = torch.cat((x,x1),dim=1)
            #time3 = time.time_ns()
            x22 = self.relu(self.dropout2(self.conv22(self.conv21(x21))))
            #time4 = time.time_ns()
            x31 = torch.cat((x21,x22),dim=1)
            #time5 = time.time_ns()
            out = self.relu(self.dropout3(self.conv32(self.conv31(x31))))
        else:
            x1 = self.relu(self.conv1(x))
            #time2 = time.time_ns()
            x21 = torch.cat((x,x1),dim=1)
            #time3 = time.time_ns()
            x22 = self.relu(self.conv22(self.conv21(x21)))
            #time4 = time.time_ns()
            x31 = torch.cat((x21,x22),dim=1)
            #time5= time.time_ns()
            out = self.relu(self.conv32(self.conv31(x31)))
        
        #time6 = time.time_ns()
        dbBn = self.bn(out)
        #time7 = time.time_ns()

        '''
        deltaTime1 = time1 - time0
        deltaTime2 = time2 - time1
        deltaTime3 = time3 - time2
        deltaTime4 = time4 - time3
        deltaTime5 = time5 - time4
        deltaTime6 = time6 - time5
        deltaTime7 = time7 - time6

        print("DownBlock " + str(deltaTime1) + ' ' + str(deltaTime2) + ' ' + str(deltaTime3) + ' ' + str(deltaTime4) + ' ' + str(deltaTime5) + ' ' + str(deltaTime6) + ' ' + str(deltaTime7))
        '''
        return dbBn
    
    
class DenseNet2D_up_block_concat(nn.Module):
    def __init__(self,skip_channels,input_channels,output_channels,up_stride,dropout=False,prob=0):
        super(DenseNet2D_up_block_concat, self).__init__()
        self.conv11 = nn.Conv2d(skip_channels+input_channels,output_channels,kernel_size=(1,1),padding=(0,0))
        self.conv12 = nn.Conv2d(output_channels,output_channels,kernel_size=(3,3),padding=(1,1))
        self.conv21 = nn.Conv2d(skip_channels+input_channels+output_channels,output_channels,
                                kernel_size=(1,1),padding=(0,0))
        self.conv22 = nn.Conv2d(output_channels,output_channels,kernel_size=(3,3),padding=(1,1))
        self.relu = nn.LeakyReLU()
        self.up_stride = up_stride
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)

    def forward(self,prev_feature_map,x):
        #time0 = time.time_ns()
        x = nn.functional.interpolate(x,scale_factor=self.up_stride,mode='nearest')
        #time1 = time.time_ns()
        x = torch.cat((x,prev_feature_map),dim=1)
        #time2 = time.time_ns()
        if self.dropout:
            x1 = self.relu(self.dropout1(self.conv12(self.conv11(x))))
            #time3 = time.time_ns()
            x21 = torch.cat((x,x1),dim=1)
            #time4 = time.time_ns()
            out = self.relu(self.dropout2(self.conv22(self.conv21(x21))))
        else:
            x1 = self.relu(self.conv12(self.conv11(x)))
            #time3 = time.time_ns()
            x21 = torch.cat((x,x1),dim=1)
            #time4 = time.time_ns()
            out = self.relu(self.conv22(self.conv21(x21)))
        #time5 = time.time_ns()

        '''
        deltaTime1 = time1 - time0
        deltaTime2 = time2 - time1
        deltaTime3 = time3 - time2
        deltaTime4 = time4 - time3
        deltaTime5 = time5 - time4

        print("UpBlock " + str(deltaTime1) + ' ' + str(deltaTime2) + ' ' + str(deltaTime3) + ' ' + str(deltaTime4) + ' ' + str(deltaTime5))
        '''

        return out
    
class DenseNet2D(nn.Module):
    def __init__(self,in_channels=1,out_channels=4,channel_size=32,concat=True,dropout=False,prob=0):
        super(DenseNet2D, self).__init__()

        self.down_block1 = DenseNet2D_down_block(input_channels=in_channels,output_channels=channel_size,
                                                 down_size=None,dropout=dropout,prob=prob)
        self.down_block2 = DenseNet2D_down_block(input_channels=channel_size,output_channels=channel_size,
                                                 down_size=(2,2),dropout=dropout,prob=prob)
        self.down_block3 = DenseNet2D_down_block(input_channels=channel_size,output_channels=channel_size,
                                                 down_size=(2,2),dropout=dropout,prob=prob)
        self.down_block4 = DenseNet2D_down_block(input_channels=channel_size,output_channels=channel_size,
                                                 down_size=(2,2),dropout=dropout,prob=prob)
        self.down_block5 = DenseNet2D_down_block(input_channels=channel_size,output_channels=channel_size,
                                                 down_size=(2,2),dropout=dropout,prob=prob)

        self.up_block1 = DenseNet2D_up_block_concat(skip_channels=channel_size,input_channels=channel_size,
                                                    output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob)
        self.up_block2 = DenseNet2D_up_block_concat(skip_channels=channel_size,input_channels=channel_size,
                                                    output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob)
        self.up_block3 = DenseNet2D_up_block_concat(skip_channels=channel_size,input_channels=channel_size,
                                                    output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob)
        self.up_block4 = DenseNet2D_up_block_concat(skip_channels=channel_size,input_channels=channel_size,
                                                    output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob)

        self.out_conv1 = nn.Conv2d(in_channels=channel_size,out_channels=out_channels,kernel_size=1,padding=0)
        self.concat = concat
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
    def forward(self,x):
        #time0 = time.time_ns()
        self.x1 = self.down_block1(x)
        #time1 = time.time_ns()
        self.x2 = self.down_block2(self.x1)
        #time2 = time.time_ns()
        self.x3 = self.down_block3(self.x2)
        #time3 = time.time_ns()
        self.x4 = self.down_block4(self.x3)
        #time4 = time.time_ns()
        self.x5 = self.down_block5(self.x4)
        #time5 = time.time_ns()
        self.x6 = self.up_block1(self.x4,self.x5)
        #time6 = time.time_ns()
        self.x7 = self.up_block2(self.x3,self.x6)
        #time7 = time.time_ns()
        self.x8 = self.up_block3(self.x2,self.x7)
        #time8 = time.time_ns()
        self.x9 = self.up_block4(self.x1,self.x8)

        #time9 = time.time_ns()
        if self.dropout:
            out = self.out_conv1(self.dropout1(self.x9))
        else:
            out = self.out_conv1(self.x9)
        
        #time10 = time.time_ns()

        #deltaTime10 = time10 - time9
        '''
        deltaTime1 = time1 - time0;
        deltaTime2 = time2 - time1;
        deltaTime3 = time3 - time2;
        deltaTime4 = time4 - time3;
        deltaTime5 = time5 - time4;
        deltaTime6 = time6 - time5;
        deltaTime7 = time7 - time6;
        deltaTime8 = time8 - time7;
        deltaTime9 = time9 - time8;
        deltaTime10 = time10 - time9;

        print(str(deltaTime1) + ' ' + str(deltaTime2) + ' ' + str(deltaTime3) + ' ' + str(deltaTime4) + ' ' + str(deltaTime5) + ' ' + str(deltaTime6) + ' ' + str(deltaTime7) + ' ' + str(deltaTime8) + ' ' + str(deltaTime9) + ' ' + str(deltaTime10))
        

        print("FinalConv2d " + str(deltaTime10))
        '''

        #deltaTime = time10 - time0
        #print(str(deltaTime))

        return out

