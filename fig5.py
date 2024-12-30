# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 14:30:14 2023

@author: huangting
"""

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
# import seaborn as sns
# sns.set(style='darkgrid')
y1=[24.82,30.14,21.61,25.23,29.54,52.43]
y2=[2.19,2.02,1.81,1.68,1.44,1.43]
names = ['1.5/1.5','3/3','6/6', '9/9', '12/12', '24/24']   
x = range(len(names))


font = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 16,}
font1 = {'family' : 'Times New Roman',
             'weight' : 'normal',
             'size'   : 14,}

plt.subplot(121)
plt.plot(x, y1, 'o-')
plt.xticks(x, names, rotation=0)
plt.xlabel('Different periodic cold shocks',font)
plt.ylabel('90% of the steady-state concentration of COR15A mRNA',font1)


plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.yticks(fontproperties = 'Times New Roman', size = 14)
    
plt.ylim(20,60)
    
y_major_locator=MultipleLocator(5)
ax=plt.gca()
ax.yaxis.set_major_locator(y_major_locator)



plt.subplot(122)
plt.plot(x, y2, 'o-')
plt.xticks(x, names, rotation=0)
plt.xlabel('Different periodic cold shocks',font)
plt.ylabel('The steady-state concentration of COR15A mRNA',font1)

plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.yticks(fontproperties = 'Times New Roman', size = 14)

plt.ylim(1,2.4)
    
y_major_locator=MultipleLocator(0.2)
ax=plt.gca()
ax.yaxis.set_major_locator(y_major_locator)


