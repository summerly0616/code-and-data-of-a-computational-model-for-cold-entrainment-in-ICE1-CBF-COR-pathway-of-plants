# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 14:30:14 2023

@author: huangting
"""

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
from scipy.interpolate import make_interp_spline
# import seaborn as sns
# sns.set(style='darkgrid')
x1=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14])
x2=np.array([6,7,8,9,10,11,12,13,14])
y1=np.array([2.1,2.12,2.15,2.19,2.24,2.32,2.43,2.57,2.67,2.45,2.04,1.63,1.25,0.895])
y2=np.array([3.26,3.35,3.62,5.24,6.29,6.61,6.59,6.53,6.473])


CBF_down = np.array([0.008749059,0.017809762,0.027199062,0.036935163,0.047037643,0.057527583,0.068427709,0.079762565,0.091558689,0.103844652,0.054857047,0.02563185,0.015135032,0.010896829])
CBF_up = np.array([0.047020385,0.10376251,0.173541098,0.261304132,0.374673765,0.525712955,0.733749367,1.028709321,1.439868097,1.887817175,2.2339289,2.474746722,2.641090738,2.755688241])
CBF_eq =np.array([0.022,0.055,0.09,0.15,0.21,0.31,0.39,0.55,0.75,0.98,1.17,1.32,1.46,1.6])


COR_down = np.array([7.51E-07,3.12E-06,7.30E-06,1.35E-05,2.20E-05,3.30E-05,4.68E-05,6.38E-05,8.45E-05,0.000110747,0.000162516,0.000276244,0.000483772,0.000787333])
COR_up = np.array([8.74E-05,0.000428556,0.001209554,0.002777319,0.005818326,0.011797578,0.024193053,0.054597798,0.355962963,1.372694636,1.975814371,2.314516932,2.555636708,2.749098777])
COR_eq =np.array([0.01,0.01,0.01,0.01,0.01,0.02,0.03,0.04,0.16,0.66,0.92,1.11,1.25,1.37])




font = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 16,}
font1 = {'family' : 'Times New Roman',
             'weight' : 'normal',
             'size'   : 14,}



plt.subplot(131)


x1_smooth = np.linspace(x1.min(), x1.max(), 5000)
CBF_down_smooth = make_interp_spline(x1, CBF_down)(x1_smooth)
CBF_up_smooth = make_interp_spline(x1, CBF_up)(x1_smooth)
CBF_eq_smooth = make_interp_spline(x1, CBF_eq)(x1_smooth)


plt.plot(x1_smooth, CBF_down_smooth,color='k')
plt.plot(x1_smooth, CBF_up_smooth,color='k')
plt.plot(x1_smooth, CBF_eq_smooth,'--',color='b')

plt.plot(11,0.054857047,'o',color='r')
plt.plot(11,2.2339289,'o',color='r')
plt.plot(11,1.17,'o',color='b')


plt.xlabel('$v_{s1}$', font)
plt.ylabel('The concentration of CBF3 mRNA',font1)


plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.yticks(fontproperties = 'Times New Roman', size = 14)

plt.xlim(0,14)    
plt.ylim(0,3)

x_major_locator=MultipleLocator(2)   
y_major_locator=MultipleLocator(1)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)



plt.subplot(132)
x1_smooth = np.linspace(x1.min(), x1.max(), 10000)
COR_down_smooth = make_interp_spline(x1, COR_down)(x1_smooth)
COR_up_smooth = make_interp_spline(x1, COR_up)(x1_smooth)
COR_eq_smooth = make_interp_spline(x1, COR_eq)(x1_smooth)


plt.plot(x1_smooth, COR_down_smooth,color='k')
plt.plot(x1_smooth, COR_up_smooth,color='k')
plt.plot(x1_smooth, COR_eq_smooth,'--',color='b')


plt.plot(11,0.000162516,'o',color='r')
plt.plot(11,1.975814371,'o',color='r')
plt.plot(11,0.92,'o',color='b')

plt.xlabel('$v_{s1}$', font)
plt.ylabel('The concentration of COR15A mRNA',font1)


plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.yticks(fontproperties = 'Times New Roman', size = 14)

plt.xlim(0,14)    
plt.ylim(0,3)

x_major_locator=MultipleLocator(2)   
y_major_locator=MultipleLocator(1)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)



plt.subplot(133)
plt.plot(x1, y1-2, 'o-',color='k',label='CBF3')

plt.plot(x2, y2-2, 'o-',color='b',label='COR15A')

plt.plot(11,0.04,'o',color='r')
plt.plot(11,4.61,'o',color='r')

plt.xlabel('$v_{s1}$', font)
plt.ylabel('Phase of transcript',font1)


plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.yticks(fontproperties = 'Times New Roman', size = 14)

plt.xlim(0,14)    
plt.ylim(-2,5)

x_major_locator=MultipleLocator(2)   
y_major_locator=MultipleLocator(1)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)

plt.legend(prop={'family' : 'Times New Roman', 'size'   : 10})



