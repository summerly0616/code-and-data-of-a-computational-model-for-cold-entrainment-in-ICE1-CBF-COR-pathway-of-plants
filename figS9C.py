# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 17:57:16 2021

@author: Huangting
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy import signal
from scipy.optimize import fsolve 


def f(y,t,vd6):
    MYB15 = y[0]
    ICE1  = y[1]
    ICE1P = y[2]
    MCBF3 = y[3]
    CBF3  = y[4]
    CBF3P = y[5]
    MZAT12= y[6]
    ZAT12 = y[7]
    MCOR15A= y[8]
    COR15A= y[9]    
    KCa = 0.14
    MYB15t = 2
    v1 = 4  
    v2 = 1.2
    v3 = 2    
    k1 = 4.92
    k2 = 2.32
    k3 = 2
    K1 = 0.5 
    K1P= 0.5 
    K2 = 0.45
    K2P= 0.8
    K3 = 0.6
    K3P= 0.58
    ks1= 0.18
    ks2= 0.25
    ks3= 1.03
    ks4= 1.2
    vs1= 20  
    vs2= 13.5  
    vs3= 1
    Ka1= 0.35
    Ka2= 0.3
    Ka3= 0.6
    KI1= 0.21
    KI2= 0.8  
    KI3= 0.3 
    KI4= 0.28
    KI5= 0.1
    KI6= 0.2
    vd1= 0.1
    vd2= 1.1
    vd3= 0.65
    vd4= 1.2
    vd5= 1.8
    vd7= 1.3
    vd8= 1.25
    vd9= 0.3
    Kd1= 0.2
    Kd2= 0.4
    Kd3= 0.6  
    Kd4= 0.1
    Kd5= 0.25
    Kd6= 0.01
    Kd7= 1.25
    Kd8= 0.75
    Kd9= 0.13
    n  = 2
    m  = 2
    r  = 2
    s  = 2
    u  = 2
    v  = 2
    n0 = 4
    Ca = 0.1
    CaM = Ca**n0/(KCa**n0+Ca**n0)
    z=np.array([ v1*(MYB15t-MYB15)/(K1P+MYB15t-MYB15)-k1*CaM*MYB15/(K1+MYB15),\
                 ks1-k2*CaM/(Ka1+CaM)*ICE1/(K2+ICE1)*Ka2/(Ka2+CaM)+v2*ICE1P/(K2P+ICE1P)-vd1*ICE1/(Kd1+ICE1),\
                 k2*CaM/(Ka1+CaM)*ICE1/(K2+ICE1)*Ka2/(Ka2+CaM)-v2*ICE1P/(K2P+ICE1P)-vd2*ICE1P/(Kd2+ICE1P)*CaM/(Ka3+CaM),\
                 vs1*CaM*ICE1**n/(KI1**n+ICE1**n)*KI2**m/(KI2**m+MYB15**m)*KI3**r/(KI3**r+ZAT12**r)-vd3*MCBF3/(Kd3+MCBF3),\
                 ks2*MCBF3-k3*CaM*CBF3/(K3+CBF3)+v3*CBF3P/(K3P+CBF3P)-vd4*CBF3/(Kd4+CBF3),\
                 k3*CaM*CBF3/(K3+CBF3)-v3*CBF3P/(K3P+CBF3P)-vd5*CBF3P/(Kd5+CBF3P),\
                 vs2*CBF3**s/(KI4**s+CBF3**s)-vd6*MZAT12/(Kd6+MZAT12),\
                 ks3*MZAT12-vd7*ZAT12/(Kd7+ZAT12),\
                 vs3*(1-1/(1+(CBF3/KI5)**u+(ZAT12/KI6)**v))-vd8*MCOR15A/(Kd8+MCOR15A),\
                 ks4*MCOR15A-vd9*COR15A/(Kd9+COR15A)])
    return z
 
def mian(vd6,T,Greater,Less):
    tspan = np.array([0,400])
    h=0.05
    t=np.arange(tspan[0],tspan[1],h)
    y0 = np.array([1.88,0.2259,0.44,0.34,0.0750,0.0059,0.51,0.01,0.08,0.1])
    yy=odeint(f,y0,t, (vd6,))
    diff=0.05*np.diff(signal.argrelextrema(yy[:,3], np.less))
    T.append(np.sum(diff[0,:]/len(diff[0,:])))
    Greater.append(yy[:,3][signal.argrelextrema(yy[:,3], np.greater)][-1])
    Less.append(yy[:,3][signal.argrelextrema(yy[:,3], np.less)][-1])   
    return (T,Greater,Less)


    
T=[]
Greater=[]   
Less=[] 
Vd6=[]
for i in range(10,3500):
    vd6=i/1000
    Vd6.append(vd6)
    mian(vd6,T,Greater,Less)
##    
##    
##    
font = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 16,}
font1 = {'family' : 'Times New Roman',
             'weight' : 'normal',
             'size'   : 14,}
 
plt.plot(Vd6,Greater,'-',color='k')
plt.plot(Vd6,Less,'-',color='k')
#
plt.xlim([0,3.2])
plt.ylim([0,3.2])
plt.xlabel('$v_{d6}$',font)
plt.ylabel('CBF3 mRNA',font1)
plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.yticks(fontproperties = 'Times New Roman', size = 14)
##plt.savefig('./vd6bifur.jpg')

x = np.arange(0.04,2.77,0.01)
y = 2.0661*x**0.3496
plt.plot(x,y,'--',color='b')