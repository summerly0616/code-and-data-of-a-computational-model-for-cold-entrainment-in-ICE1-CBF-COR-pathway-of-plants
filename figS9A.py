# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 11:32:18 2021

@author: Huangting
"""

import numpy as np
import matplotlib.pyplot as plt

font = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 16,}
font1 = {'family' : 'Times New Roman',
             'weight' : 'normal',
             'size'   : 14,}
font2 = {'family' : 'Times New Roman',
             'weight' : 'normal',
             'size'   : 12,}


vd6 =    [0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.20,0.21,0.22,0.3,0.4,0.5,0.6,0.7,1,2]
ks3low = [1.57,1.08,0.86,0.49,0.4,0.31,0.19,0.13,0.13,0.12,0.12,0.11,0.11,0.11,0.09,0.08,0.069,0.069,0.069,0.069,0.053,0.029,0.029,0.029,0.02,0.02,0.02]
ks3up  = [68.93,48.22,36.9,30.23,25.82,22.64,20.28,18.23,16.91,15.53,14.56,13.64,12.94,12.23,11.71,11.15,10.71,10.22,10,9.88,7.61,6.33,5.53,4.98,4.57,3.79,2.97]
plt.plot(vd6,ks3low,'-',color='k')
plt.plot(vd6,ks3up,'-',color='k')
plt.fill(np.append(vd6, vd6[::-1]), np.append(ks3low, ks3up[::-1]), 'lightgrey')
plt.xlim([0,2])
plt.ylim([0,70])
plt.xlabel('$v_{d6}$',font)
plt.ylabel('$k_{s3}$',font1)
plt.text(0.033, 3.15,'Sustained Oscillations',color='k',fontdict=font2)
plt.text(0.18, 52.88,'Steady State',color='k',fontdict=font1)
plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.yticks(fontproperties = 'Times New Roman', size = 14)