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
x1=np.array([1,1.5,2,2.5,3,3.5,4,5,6])

y1=np.array([0.44,2.07,2.5,2.245,2.145,2.1,2.075,2.05,2.035])
y2=np.array([6.385,6.615,3.755,3.2,3.13,3.11,3.1,3.09,3.085])


CBF_down = np.array([0.005680055,0.055891798,0.082666971,0.064016626,0.052232555,0.044112428,0.038177337,0.030082472,0.024819843])
CBF_up = np.array([2.832376789,2.257190782,1.148652152,0.650318754,0.447669452,0.34065038,0.274806009,0.1981331,0.154897057])
CBF_eq =np.array([1.419028422,1.15654129,0.615659562,0.35716769,0.249951003,0.192381404,0.156491673,0.114107786,0.08985845])


COR_down = np.array([3.14E-03,1.62E-04,6.86E-05,4.09E-05,2.71E-05,1.93E-05,1.44E-05,8.94E-06,6.07E-06])
COR_up = np.array([3.016432406,1.96015206,0.075246944,0.018447417,0.008385473,0.004768993,0.003072911,0.001580594,0.000960721])
COR_eq =np.array([1.509787512,0.980156978,0.03765779,0.009244167,0.004206305,0.00239415,0.001543673,0.000794767,0.000483398])




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

plt.plot(1.5,0.055891798,'o',color='r')
plt.plot(1.5,2.257190782,'o',color='r')
plt.plot(1.5,1.15654129,'o',color='b')


plt.xlabel('$v_{d3}$', font)
plt.ylabel('The concentration of CBF3 mRNA',font1)


plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.yticks(fontproperties = 'Times New Roman', size = 14)

plt.xlim(0.5,6)    
plt.ylim(0,3)

x_major_locator=MultipleLocator(1)   
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


plt.plot(1.5,0.000161895,'o',color='r')
plt.plot(1.5,1.96015206,'o',color='r')
plt.plot(1.5,0.980156978,'o',color='b')

plt.xlabel('$v_{d3}$', font)
plt.ylabel('The concentration of COR15A mRNA',font1)


plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.yticks(fontproperties = 'Times New Roman', size = 14)

plt.xlim(1,2)    
plt.ylim(0,3)

x_major_locator=MultipleLocator(0.5)   
y_major_locator=MultipleLocator(1)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)



plt.subplot(133)
plt.plot(x1, y1-2, 'o-',color='k',label='CBF3')

plt.plot(x1, y2-2, 'o-',color='b',label='COR15A')

plt.plot(1.5,0.07,'o',color='r')
plt.plot(1.5,4.615,'o',color='r')

plt.xlabel('$v_{d3}$', font)
plt.ylabel('Phase of transcript',font1)


plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.yticks(fontproperties = 'Times New Roman', size = 14)

plt.xlim(0,6)    
plt.ylim(-2,5)

x_major_locator=MultipleLocator(2)   
y_major_locator=MultipleLocator(1)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)

plt.legend(prop={'family' : 'Times New Roman', 'size'   : 10})



