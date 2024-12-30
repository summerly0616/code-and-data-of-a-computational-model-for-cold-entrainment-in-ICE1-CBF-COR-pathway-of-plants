# -*- coding: utf-8 -*-
"""
Created on Fri May 21 20:09:50 2021

@author: Huangtin
"""

import pandas as pd
import matplotlib.pyplot as plt

# 设置中文显示
import matplotlib as mpl
mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False

# 使用 pandas 读取csv文件
file1 = pd.read_csv('sen_amplitude.csv', header = None, sep = ',') 
df1 = pd.DataFrame(file1)


names = ('$KCa$','$v_{1}$','$v_{2}$','$v_{3}$','$k_{1}$','$k_{2}$','$k_{3}$','$K_{1}$','$K_{1P}$','$K_{2}$','$K_{2P}$','$K_{3}$','$K_{3P}$','$k_{s1}$','$k_{s2}$','$k_{s3}$','$k_{s4}$','$v_{s1}$','$v_{s2}$','$v_{s3}$','$K_{a1}$','$K_{a2}$','$K_{a3}$','$K_{I1}$','$K_{I2}$','$K_{I3}$','$K_{I4}$','$K_{I5}$','$K_{I6}$','$v_{d1}$','$v_{d2}$','$v_{d3w}$','$v_{d3c}$','$v_{d4}$','$v_{d5}$','$v_{d6}$','$v_{d7}$','$v_{d8}$','$v_{d9}$','$K_{d1}$','$K_{d2}$','$K_{d3}$','$K_{d4}$','$K_{d5}$','$K_{d6}$','$K_{d7}$','$K_{d8}$','$K_{d9}$')
v11 = df1[1]	
v12 = df1[2]
font = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 16,}
font1 = {'family' : 'Times New Roman',
             'weight' : 'normal',
             'size'   : 14,}
font2 = {'family' : 'Times New Roman',
             'weight' : 'normal',
             'size'   : 12,}

plt.subplot(131)
plt.barh(names, v11, color = 'darkblue')	
plt.barh(names, v12, color = 'darkblue')	
plt.vlines(0,-5, 50,color='darkred',linewidth=2)
plt.xlim([-3,3])
plt.ylim([-0.2,47.5])
plt.xlabel('fold change(log$_{10}$)',font)
plt.ylabel('Parameters',font1)
plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.title('Sensitivity analysis for the amplitude of CBF3 response',font2)

file2 = pd.read_csv('sen_phase.csv', header = None, sep = ',') 
df2 = pd.DataFrame(file2)
names = ('$KCa$','$v_{1}$','$v_{2}$','$v_{3}$','$k_{1}$','$k_{2}$','$k_{3}$','$K_{1}$','$K_{1P}$','$K_{2}$','$K_{2P}$','$K_{3}$','$K_{3P}$','$k_{s1}$','$k_{s2}$','$k_{s3}$','$k_{s4}$','$v_{s1}$','$v_{s2}$','$v_{s3}$','$K_{a1}$','$K_{a2}$','$K_{a3}$','$K_{I1}$','$K_{I2}$','$K_{I3}$','$K_{I4}$','$K_{I5}$','$K_{I6}$','$v_{d1}$','$v_{d2}$','$v_{d3w}$','$v_{d3c}$','$v_{d4}$','$v_{d5}$','$v_{d6}$','$v_{d7}$','$v_{d8}$','$v_{d9}$','$K_{d1}$','$K_{d2}$','$K_{d3}$','$K_{d4}$','$K_{d5}$','$K_{d6}$','$K_{d7}$','$K_{d8}$','$K_{d9}$')
v21 = df2[1]	
v22 = df2[2]


plt.subplot(132)
plt.barh(names, v21, color = 'darkblue')	
plt.barh(names, v22, color = 'darkblue')	
plt.vlines(0,-5, 50,color='darkred',linewidth=2)
plt.xlim([-3,3])
plt.ylim([-0.2,47.5])
plt.xlabel('fold change(log$_{10}$)',font)
plt.ylabel('Parameters',font1)
plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.title('Sensitivity analysis for the phase of CBF3 response',font2)


file3 = pd.read_csv('sen_halfwidth.csv', header = None, sep = ',') 
df3 = pd.DataFrame(file3)
names = ('$KCa$','$v_{1}$','$v_{2}$','$v_{3}$','$k_{1}$','$k_{2}$','$k_{3}$','$K_{1}$','$K_{1P}$','$K_{2}$','$K_{2P}$','$K_{3}$','$K_{3P}$','$k_{s1}$','$k_{s2}$','$k_{s3}$','$k_{s4}$','$v_{s1}$','$v_{s2}$','$v_{s3}$','$K_{a1}$','$K_{a2}$','$K_{a3}$','$K_{I1}$','$K_{I2}$','$K_{I3}$','$K_{I4}$','$K_{I5}$','$K_{I6}$','$v_{d1}$','$v_{d2}$','$v_{d3w}$','$v_{d3c}$','$v_{d4}$','$v_{d5}$','$v_{d6}$','$v_{d7}$','$v_{d8}$','$v_{d9}$','$K_{d1}$','$K_{d2}$','$K_{d3}$','$K_{d4}$','$K_{d5}$','$K_{d6}$','$K_{d7}$','$K_{d8}$','$K_{d9}$')
v31 = df3[1]	
v32 = df3[2]

plt.subplot(133)
plt.barh(names, v31, color = 'darkblue')	
plt.barh(names, v32, color = 'darkblue')	
plt.vlines(0,-5, 50,color='darkred',linewidth=2)
plt.xlim([-3,3])
plt.ylim([-0.2,47.5])
plt.xlabel('fold change(log$_{10}$)',font)
plt.ylabel('Parameters',font1)
plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.title('Sensitivity analysis for the half peak-width of CBF3 response',font2)

plt.show()