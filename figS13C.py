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
x1=np.array([0.03,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1])

y1=np.array([2.505,2.615,2.775,2.865,2.635,2.34,2.07,1.83,1.405,1.035])
y2=np.array([3.195,3.335,3.595,5.435,6.33,6.58,6.615,6.59,6.515,6.44])


CBF_down = np.array([0.006872472,0.021665088,0.033330184,0.04666043,0.06999071,0.069658696,0.055891798,0.050069135,0.048698356,0.054510936])
CBF_up = np.array([0.397455546,0.492547854,0.990055662,1.483625896,1.945847021,2.144467336,2.257190782,2.322576106,2.378247466,2.382649014])
CBF_eq =np.array([0.202164009,0.302106471,0.506692923,0.815143163,1.007918866,1.107063016,1.15654129,1.186322621,1.213472911,1.218579975])


COR_down = np.array([4.81E-07,1.34E-06,5.37E-06,2.16E-05,5.02E-05,9.59E-05,0.000161895,0.000247581,0.000466033,0.000725832])
COR_up = np.array([0.004899353,0.012041323,0.040974925,0.447448104,1.284907267,1.712283166,1.96015206,2.135342407,2.389229384,2.579046631])
COR_eq =np.array([0.002449917,0.00602133,0.020490147,0.223734875,0.642478732,0.856189536,0.980156978,1.067794994,1.194847709,1.289886232])




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

plt.plot(0.5,0.055891798,'o',color='r')
plt.plot(0.5,2.257190782,'o',color='r')
plt.plot(0.5,1.15654129,'o',color='b')


plt.xlabel('$K_{d3}$', font)
plt.ylabel('The concentration of CBF3 mRNA',font1)


plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.yticks(fontproperties = 'Times New Roman', size = 14)

plt.xlim(0,1)    
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


plt.plot(0.5,0.000161895,'o',color='r')
plt.plot(0.5,1.96015206,'o',color='r')
plt.plot(0.5,0.980156978,'o',color='b')

plt.xlabel('$K_{d3}$', font)
plt.ylabel('The concentration of COR15A mRNA',font1)


plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.yticks(fontproperties = 'Times New Roman', size = 14)

plt.xlim(0,1)    
plt.ylim(0,3)

x_major_locator=MultipleLocator(0.5)   
y_major_locator=MultipleLocator(1)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)



plt.subplot(133)
plt.plot(x1, y1-2, 'o-',color='k',label='CBF3')

plt.plot(x1, y2-2, 'o-',color='b',label='COR15A')

plt.plot(0.5,0.07,'o',color='r')
plt.plot(0.5,4.615,'o',color='r')

plt.xlabel('$K_{d3}$', font)
plt.ylabel('Phase of transcript',font1)


plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.yticks(fontproperties = 'Times New Roman', size = 14)

plt.xlim(0,1)    
plt.ylim(-2,5)

x_major_locator=MultipleLocator(2)   
y_major_locator=MultipleLocator(1)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)

plt.legend(prop={'family' : 'Times New Roman', 'size'   : 10})



