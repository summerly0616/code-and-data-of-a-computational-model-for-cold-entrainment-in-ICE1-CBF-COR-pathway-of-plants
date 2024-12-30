# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 10:50:57 2022

@author: Huangtin
"""

from matplotlib.pyplot import MultipleLocator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False

#%% 使用 pandas 读取csv文件
file1 = pd.read_csv('cold_tolerance.csv', header = None, sep = ',') 
df1 = pd.DataFrame(file1)

file2 = pd.read_csv('cold_tolerance_v1.csv', header = None, sep = ',') 
df2 = pd.DataFrame(file2)

file3 = pd.read_csv('cold_tolerance_v2.csv', header = None, sep = ',') 
df3 = pd.DataFrame(file3)

file4 = pd.read_csv('cold_tolerance_v3.csv', header = None, sep = ',') 
df4 = pd.DataFrame(file4)

file5 = pd.read_csv('cold_tolerance_k1.csv', header = None, sep = ',') 
df5 = pd.DataFrame(file5)

file6 = pd.read_csv('cold_tolerance_k2.csv', header = None, sep = ',') 
df6 = pd.DataFrame(file6)

file7 = pd.read_csv('cold_tolerance_k3.csv', header = None, sep = ',') 
df7 = pd.DataFrame(file7)

file8 = pd.read_csv('cold_tolerance_KK1.csv', header = None, sep = ',') 
df8 = pd.DataFrame(file8)

file9 = pd.read_csv('cold_tolerance_KK1P.csv', header = None, sep = ',') 
df9 = pd.DataFrame(file9)

file10 = pd.read_csv('cold_tolerance_KK2.csv', header = None, sep = ',') 
df10 = pd.DataFrame(file10)

file11 = pd.read_csv('cold_tolerance_KK2P.csv', header = None, sep = ',') 
df11 = pd.DataFrame(file11)


file12 = pd.read_csv('cold_tolerance_KK3.csv', header = None, sep = ',') 
df12 = pd.DataFrame(file12)

file13 = pd.read_csv('cold_tolerance_KK3P.csv', header = None, sep = ',') 
df13 = pd.DataFrame(file13)

file14 = pd.read_csv('cold_tolerance_ks1.csv', header = None, sep = ',') 
df14 = pd.DataFrame(file14)


file15 = pd.read_csv('cold_tolerance_ks2.csv', header = None, sep = ',') 
df15 = pd.DataFrame(file15)


file16 = pd.read_csv('cold_tolerance_ks3.csv', header = None, sep = ',') 
df16 = pd.DataFrame(file16)

file17 = pd.read_csv('cold_tolerance_ks4.csv', header = None, sep = ',') 
df17 = pd.DataFrame(file17)

file18 = pd.read_csv('cold_tolerance_vs2.csv', header = None, sep = ',') 
df18 = pd.DataFrame(file18)

file19 = pd.read_csv('cold_tolerance_Ka1.csv', header = None, sep = ',') 
df19 = pd.DataFrame(file19)

file20 = pd.read_csv('cold_tolerance_Ka2.csv', header = None, sep = ',') 
df20 = pd.DataFrame(file20)

file21 = pd.read_csv('cold_tolerance_Ka3.csv', header = None, sep = ',') 
df21 = pd.DataFrame(file21)


file22 = pd.read_csv('cold_tolerance_Ka4.csv', header = None, sep = ',') 
df22 = pd.DataFrame(file22)

file23 = pd.read_csv('cold_tolerance_vd1.csv', header = None, sep = ',') 
df23 = pd.DataFrame(file23)

file24 = pd.read_csv('cold_tolerance_vd2.csv', header = None, sep = ',') 
df24 = pd.DataFrame(file24)


file25 = pd.read_csv('cold_tolerance_vd3c.csv', header = None, sep = ',') 
df25 = pd.DataFrame(file25)

file26 = pd.read_csv('cold_tolerance_vd3w.csv', header = None, sep = ',') 
df26 = pd.DataFrame(file26)


file27 = pd.read_csv('cold_tolerance_vd4.csv', header = None, sep = ',') 
df27 = pd.DataFrame(file27)

file28 = pd.read_csv('cold_tolerance_vd5.csv', header = None, sep = ',') 
df28 = pd.DataFrame(file28)

file29 = pd.read_csv('cold_tolerance_vd6.csv', header = None, sep = ',') 
df29 = pd.DataFrame(file29)

file30 = pd.read_csv('cold_tolerance_vd7.csv', header = None, sep = ',') 
df30 = pd.DataFrame(file30)

file31 = pd.read_csv('cold_tolerance_vd8.csv', header = None, sep = ',') 
df31 = pd.DataFrame(file31)

file32 = pd.read_csv('cold_tolerance_vd9.csv', header = None, sep = ',') 
df32 = pd.DataFrame(file32)

file33 = pd.read_csv('cold_tolerance_KI1.csv', header = None, sep = ',') 
df33 = pd.DataFrame(file33)

file34 = pd.read_csv('cold_tolerance_KI2.csv', header = None, sep = ',') 
df34 = pd.DataFrame(file34)

file35 = pd.read_csv('cold_tolerance_KI3.csv', header = None, sep = ',') 
df35 = pd.DataFrame(file35)

file36 = pd.read_csv('cold_tolerance_KI4.csv', header = None, sep = ',') 
df36 = pd.DataFrame(file36)

file37 = pd.read_csv('cold_tolerance_KI5.csv', header = None, sep = ',') 
df37 = pd.DataFrame(file37)

file38 = pd.read_csv('cold_tolerance_KI6.csv', header = None, sep = ',') 
df38 = pd.DataFrame(file38)

file39 = pd.read_csv('cold_tolerance_Kd1.csv', header = None, sep = ',') 
df39 = pd.DataFrame(file39)

file40 = pd.read_csv('cold_tolerance_Kd2.csv', header = None, sep = ',') 
df40 = pd.DataFrame(file40)

file41 = pd.read_csv('cold_tolerance_Kd3.csv', header = None, sep = ',') 
df41 = pd.DataFrame(file41)

file42 = pd.read_csv('cold_tolerance_Kd4.csv', header = None, sep = ',') 
df42 = pd.DataFrame(file42)

file43 = pd.read_csv('cold_tolerance_Kd5.csv', header = None, sep = ',') 
df43 = pd.DataFrame(file43)

file44 = pd.read_csv('cold_tolerance_Kd6.csv', header = None, sep = ',') 
df44 = pd.DataFrame(file44)

file45 = pd.read_csv('cold_tolerance_Kd7.csv', header = None, sep = ',') 
df45 = pd.DataFrame(file45)

file46 = pd.read_csv('cold_tolerance_Kd8.csv', header = None, sep = ',') 
df46 = pd.DataFrame(file46)

file47 = pd.read_csv('cold_tolerance_Kd9.csv', header = None, sep = ',') 
df47 = pd.DataFrame(file47)



par1=df1[0]
tt1=df1[1]

par2=df2[0]
tt2=df2[1]

par3=df3[0]
tt3=df3[1]

par4=df4[0]
tt4=df4[1]

par5=df5[0]
tt5=df5[1]

par6=df6[0]
tt6=df6[1]

par7=df7[0]
tt7=df7[1]

par8=df8[0]
tt8=df8[1]


par9=df9[0]
tt9=df9[1]

par10=df10[0]
tt10=df10[1]

par11=df11[0]
tt11=df11[1]

par12=df12[0]
tt12=df12[1]

par13=df13[0]
tt13=df13[1]

par14=df14[0]
tt14=df14[1]

par15=df15[0]
tt15=df15[1]

par16=df16[0]
tt16=df16[1]

par17=df17[0]
tt17=df17[1]

par18=df18[0]
tt18=df18[1]

par19=df19[0]
tt19=df19[1]

par20=df20[0]
tt20=df20[1]

par21=df21[0]
tt21=df21[1]

par22=df22[0]
tt22=df22[1]


par23=df23[0]
tt23=df23[1]

par24=df24[0]
tt24=df24[1]

par25=df25[0]
tt25=df25[1]

par26=df26[0]
tt26=df26[1]

par27=df27[0]
tt27=df27[1]

par28=df28[0]
tt28=df28[1]


par29=df29[0]
tt29=df29[1]

par30=df30[0]
tt30=df30[1]

par31=df31[0]
tt31=df31[1]

par32=df32[0]
tt32=df32[1]

par33=df33[0]
tt33=df33[1]

par34=df34[0]
tt34=df34[1]

par35=df35[0]
tt35=df35[1]

par36=df36[0]
tt36=df36[1]

par37=df37[0]
tt37=df37[1]


par38=df38[0]
tt38=df38[1]

par39=df39[0]
tt39=df39[1]

par40=df40[0]
tt40=df40[1]

par41=df41[0]
tt41=df41[1]

par42=df42[0]
tt42=df42[1]

par43=df43[0]
tt43=df43[1]

par44=df44[0]
tt44=df44[1]

par45=df45[0]
tt45=df45[1]

par46=df46[0]
tt46=df46[1]

par47=df47[0]
tt47=df47[1]

#BuPu
font = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 16,}
font1 = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 14,}
x1 = df1[0]
yy2 = 1*np.ones(len(x1))
col = np.array(df1[1])
sc1 = plt.scatter(df1[0],yy2,s=80,marker='s',c=col,cmap='plasma')
plt.colorbar(sc1)

x1 = df2[0]
yy2 = 1.1*np.ones(len(x1))
col = np.array(df2[1])
sc1 = plt.scatter(df2[0],yy2,s=80,marker='s',c=col,cmap='plasma')
#plt.colorbar(sc1)

x1 = df3[0]
yy2 = 1.2*np.ones(len(x1))
col = np.array(df3[1])
sc1 = plt.scatter(df3[0],yy2,s=80,marker='s',c=col,cmap='plasma')
#plt.colorbar(sc1)


x1 = df4[0]
yy2 = 1.3*np.ones(len(x1))
col = np.array(df4[1])
sc1 = plt.scatter(df4[0],yy2,s=80,marker='s',c=col,cmap='plasma')
#plt.colorbar(sc1)

x1 = df5[0]
yy2 = 1.4*np.ones(len(x1))
col = np.array(df5[1])
sc1 = plt.scatter(df5[0],yy2,s=80,marker='s',c=col,cmap='plasma')

x1 = df6[0]
yy2 = 1.5*np.ones(len(x1))
col = np.array(df6[1])
sc1 = plt.scatter(df6[0],yy2,s=80,marker='s',c=col,cmap='plasma')

x1 = df7[0]
yy2 = 1.6*np.ones(len(x1))
col = np.array(df7[1])
sc1 = plt.scatter(df7[0],yy2,s=80,marker='s',c=col,cmap='plasma')

x1 = df8[0]
yy2 = 1.7*np.ones(len(x1))
col = np.array(df8[1])
sc1 = plt.scatter(df8[0],yy2,s=80,marker='s',c=col,cmap='plasma')

x1 = df9[0]
yy2 = 1.8*np.ones(len(x1))
col = np.array(df9[1])
sc1 = plt.scatter(df9[0],yy2,s=80,marker='s',c=col,cmap='plasma')

x1 = df10[0]
yy2 = 1.9*np.ones(len(x1))
col = np.array(df10[1])
sc1 = plt.scatter(df10[0],yy2,s=80,marker='s',c=col,cmap='plasma')

x1 = df11[0]
yy2 = 2*np.ones(len(x1))
col = np.array(df11[1])
sc1 = plt.scatter(df11[0],yy2,s=80,marker='s',c=col,cmap='plasma')

x1 = df12[0]
yy2 = 2.1*np.ones(len(x1))
col = np.array(df12[1])
sc1 = plt.scatter(df12[0],yy2,s=80,marker='s',c=col,cmap='plasma')


x1 = df13[0]
yy2 = 2.2*np.ones(len(x1))
col = np.array(df13[1])
sc1 = plt.scatter(df13[0],yy2,s=80,marker='s',c=col,cmap='plasma')

x1 = df14[0]
yy2 = 2.3*np.ones(len(x1))
col = np.array(df14[1])
sc1 = plt.scatter(df14[0],yy2,s=80,marker='s',c=col,cmap='plasma')

x1 = df15[0]
yy2 = 2.4*np.ones(len(x1))
col = np.array(df15[1])
sc1 = plt.scatter(df15[0],yy2,s=80,marker='s',c=col,cmap='plasma')

x1 = df16[0]
yy2 = 2.5*np.ones(len(x1))
col = np.array(df16[1])
sc1 = plt.scatter(df16[0],yy2,s=80,marker='s',c=col,cmap='plasma')

x1 = df17[0]
yy2 = 2.6*np.ones(len(x1))
col = np.array(df17[1])
sc1 = plt.scatter(df17[0],yy2,s=80,marker='s',c=col,cmap='plasma')


x1 = df18[0]
yy2 = 2.7*np.ones(len(x1))
col = np.array(df18[1])
sc1 = plt.scatter(df18[0],yy2,s=80,marker='s',c=col,cmap='plasma')


x1 = df19[0]
yy2 = 2.8*np.ones(len(x1))
col = np.array(df19[1])
sc1 = plt.scatter(df19[0],yy2,s=80,marker='s',c=col,cmap='plasma')

x1 = df20[0]
yy2 = 2.9*np.ones(len(x1))
col = np.array(df20[1])
sc1 = plt.scatter(df20[0],yy2,s=80,marker='s',c=col,cmap='plasma')

x1 = df21[0]
yy2 = 3*np.ones(len(x1))
col = np.array(df21[1])
sc1 = plt.scatter(df21[0],yy2,s=80,marker='s',c=col,cmap='plasma')

x1 = df22[0]
yy2 = 3.1*np.ones(len(x1))
col = np.array(df22[1])
sc1 = plt.scatter(df22[0],yy2,s=80,marker='s',c=col,cmap='plasma')

x1 = df23[0]
yy2 = 3.2*np.ones(len(x1))
col = np.array(df23[1])
sc1 = plt.scatter(df23[0],yy2,s=80,marker='s',c=col,cmap='plasma')

x1 = df24[0]
yy2 = 3.3*np.ones(len(x1))
col = np.array(df24[1])
sc1 = plt.scatter(df24[0],yy2,s=80,marker='s',c=col,cmap='plasma')

x1 = df25[0]
yy2 = 3.4*np.ones(len(x1))
col = np.array(df25[1])
sc1 = plt.scatter(df25[0],yy2,s=80,marker='s',c=col,cmap='plasma')

x1 = df26[0]
yy2 = 3.5*np.ones(len(x1))
col = np.array(df26[1])
sc1 = plt.scatter(df26[0],yy2,s=80,marker='s',c=col,cmap='plasma')

x1 = df27[0]
yy2 = 3.6*np.ones(len(x1))
col = np.array(df27[1])
sc1 = plt.scatter(df27[0],yy2,s=80,marker='s',c=col,cmap='plasma')

x1 = df28[0]
yy2 = 3.7*np.ones(len(x1))
col = np.array(df28[1])
sc1 = plt.scatter(df28[0],yy2,s=80,marker='s',c=col,cmap='plasma')

x1 = df29[0]
yy2 = 3.8*np.ones(len(x1))
col = np.array(df29[1])
sc1 = plt.scatter(df29[0],yy2,s=80,marker='s',c=col,cmap='plasma')

x1 = df30[0]
yy2 = 3.9*np.ones(len(x1))
col = np.array(df30[1])
sc1 = plt.scatter(df30[0],yy2,s=80,marker='s',c=col,cmap='plasma')

x1 = df31[0]
yy2 = 4*np.ones(len(x1))
col = np.array(df31[1])
sc1 = plt.scatter(df31[0],yy2,s=80,marker='s',c=col,cmap='plasma')

x1 = df32[0]
yy2 = 4.1*np.ones(len(x1))
col = np.array(df32[1])
sc1 = plt.scatter(df32[0],yy2,s=80,marker='s',c=col,cmap='plasma')

x1 = df33[0]
yy2 = 4.2*np.ones(len(x1))
col = np.array(df33[1])
sc1 = plt.scatter(df33[0],yy2,s=80,marker='s',c=col,cmap='plasma')

x1 = df34[0]
yy2 = 4.3*np.ones(len(x1))
col = np.array(df34[1])
sc1 = plt.scatter(df34[0],yy2,s=80,marker='s',c=col,cmap='plasma')


x1 = df35[0]
yy2 = 4.4*np.ones(len(x1))
col = np.array(df35[1])
sc1 = plt.scatter(df35[0],yy2,s=80,marker='s',c=col,cmap='plasma')


x1 = df36[0]
yy2 = 4.5*np.ones(len(x1))
col = np.array(df36[1])
sc1 = plt.scatter(df36[0],yy2,s=80,marker='s',c=col,cmap='plasma')


x1 = df37[0]
yy2 = 4.6*np.ones(len(x1))
col = np.array(df37[1])
sc1 = plt.scatter(df37[0],yy2,s=80,marker='s',c=col,cmap='plasma')


x1 = df38[0]
yy2 = 4.7*np.ones(len(x1))
col = np.array(df38[1])
sc1 = plt.scatter(df38[0],yy2,s=80,marker='s',c=col,cmap='plasma')

x1 = df39[0]
yy2 = 4.8*np.ones(len(x1))
col = np.array(df39[1])
sc1 = plt.scatter(df39[0],yy2,s=80,marker='s',c=col,cmap='plasma')

x1 = df40[0]
yy2 = 4.9*np.ones(len(x1))
col = np.array(df40[1])
sc1 = plt.scatter(df40[0],yy2,s=80,marker='s',c=col,cmap='plasma')

x1 = df41[0]
yy2 = 5*np.ones(len(x1))
col = np.array(df41[1])
sc1 = plt.scatter(df41[0],yy2,s=80,marker='s',c=col,cmap='plasma')

x1 = df42[0]
yy2 = 5.1*np.ones(len(x1))
col = np.array(df42[1])
sc1 = plt.scatter(df42[0],yy2,s=80,marker='s',c=col,cmap='plasma')

x1 = df43[0]
yy2 = 5.2*np.ones(len(x1))
col = np.array(df43[1])
sc1 = plt.scatter(df43[0],yy2,s=80,marker='s',c=col,cmap='plasma')


x1 = df44[0]
yy2 = 5.3*np.ones(len(x1))
col = np.array(df44[1])
sc1 = plt.scatter(df44[0],yy2,s=80,marker='s',c=col,cmap='plasma')


x1 = df45[0]
yy2 = 5.4*np.ones(len(x1))
col = np.array(df45[1])
sc1 = plt.scatter(df45[0],yy2,s=80,marker='s',c=col,cmap='plasma')


x1 = df46[0]
yy2 = 5.5*np.ones(len(x1))
col = np.array(df46[1])
sc1 = plt.scatter(df46[0],yy2,s=80,marker='s',c=col,cmap='plasma')


x1 = df47[0]
yy2 = 5.6*np.ones(len(x1))
col = np.array(df47[1])
sc1 = plt.scatter(df47[0],yy2,s=80,marker='s',c=col,cmap='plasma')


font = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 18,}



plt.xlabel('Range of parameters',font)
plt.xticks(fontproperties = 'Times New Roman', size = 18)
plt.yticks(fontproperties = 'Times New Roman', size = 18)
plt.xlim(0,20)
plt.ylim(0.95,5.65)
x_major_locator=MultipleLocator(5)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
