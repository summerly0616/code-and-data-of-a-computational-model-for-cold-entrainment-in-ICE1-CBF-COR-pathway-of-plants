# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 17:27:09 2020

@author: Huangting
"""

import numpy as np
import matplotlib.pyplot as plt
#from RK4 import RK4
from Gauss2s4 import Gauss2s4
#from Gauss3s6 import Gauss3s6
 

y0 = np.array([0.1,1.92,0.2259,0.44,0.05,0.0750,0.0059,0.01,0,0.01,0.58])
h=0.0005
tspan = np.array([0,24])
t,y = Gauss2s4(tspan, y0 ,h)


#plotting time courses
font = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 16,
}

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 12,
}


plt.plot(t,y[0,:]/(max(y[0,:])),label='Ca$^{2+}$',linewidth=1.9)
plt.text(2.54, 0.17,'Ca$^{2+}$',font1)
#plt.legend()

plt.plot(t,y[1,:]/(max(y[1,:])),label='MYB15',linewidth=1.9)
plt.text(3.62, 0.79,'MYB15',font1)
#plt.legend()



plt.plot(t,y[2,:]/(max(y[2,:])),label='ICE1',linewidth=1.9)
plt.text(3.14, 0.42,'ICE1',font1)
#plt.legend()


plt.plot(t,y[4,:]/(max(y[4,:])),label='CBF3 mRNA',linewidth=1.9)
plt.text(3.30, 1.04,'MCBF3',font1)
#plt.legend()


plt.plot(t,y[7,:]/(max(y[7,:])),label='ZAT12 mRNA',linewidth=1.9)
plt.text(4.05, 0.58,'MZAT12',font1)
#plt.legend()

plt.plot(t,y[9,:]/(max(y[9,:])),label='COR15A mRNA',linewidth=1.9)
plt.text(4.28, 0.18,'MCOR15A',font1)

plt.xlim([1.5,6])
plt.ylim([-0.1,1.2])
plt.xlabel('Time(h)',font)
plt.ylabel('Relative levels',font)
plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.yticks(fontproperties = 'Times New Roman', size = 14)

plt.show()
