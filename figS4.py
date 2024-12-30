# -*- coding: utf-8 -*-
"""
Created on Tue May 25 18:15:56 2021

@author: Huangtin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint 
from scipy.signal import find_peaks

font = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 18,}
font1 = {'family' : 'Times New Roman',
             'weight' : 'normal',
             'size'   : 14,}
font2 = {'family' : 'Times New Roman',
             'weight' : 'normal',
             'size'   : 12,}

plt.subplot(131)
beta = [0.28,0.29,0.3,0.31,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
alow = np.ones((len(beta),1))*0.01
aup  = [0.01,0.56,0.82,1.09,2.34,4.15,6.22,8.54,11.05,13.7,16,17.6,19.1,20.6,22.1,23.5,24.9,26.2]
plt.plot(beta,alow,'-',color='k')
plt.plot(beta,aup,'-',color='k')
plt.fill(np.append(beta, beta[::-1]), np.append(alow, aup[::-1]), 'lightgrey')
plt.xlim([0.28,1])
plt.ylim([0,30])
plt.ylabel('a',font,style='italic')
plt.xlabel(r'$\beta_{f}$',font1)
plt.text(0.54, 4.15,'multi-spike oscillation',color='k',fontdict=font1)
plt.text(0.48, 20.88,'single spike',color='k',fontdict=font1)
plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.yticks(fontproperties = 'Times New Roman', size = 14)



def fun(y,t,betaf,a):
    KCa = 0.14
    V0 = 1*25
    V1 = 7.3*25
    kf = 1*25
    K = 10*25
    VM2 = 65*25
    KM2 = 1
    VM3 = 500*25
    KM3 = 2
    KA = 0.9
    M = 2
    N = 2
    P = 4  
    tp = 2
    T = 20*np.exp(-a*(t-tp))*(t>tp)
    beta = betaf/20*T   
    Z     = y[0]
    MYB15 = y[1]
    ICE1  = y[2]
    ICE1P = y[3]
    MCBF3 = y[4]
    CBF3  = y[5]
    CBF3P = y[6]
    MZAT12= y[7]
    ZAT12 = y[8]
    Y     = y[9]
    MCOR15A= y[10]
    COR15A= y[11]   
    V2 = VM2*Z**N/(KM2**N+Z**N)
    V3 = VM3*Y**M/(KM3**M+Y**M)*Z**P/(KA**P+Z**P)   
    Tc = 2    
    MYB15t = 2           
    v1 = 4  
    v2 = 1.2
    v3 = 2    
    k1 = 4.92
    k2 = 15
    k3 = 2
    K1 = 0.5 
    K1P= 0.5 
    K2 = 0.45
    K2P= 0.8
    K3 = 0.6
    K3P= 0.58
    ks1= 0.18
    ks2= 0.25
    ks3= 2.23
    ks4= 1.2
    vs1= 34  
    vs2= 9.5  
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
    vd3= (t-Tc>0)*0.55+(Tc-t>0)*5.5
    vd4= 1.2
    vd5= 1.8
    vd6= 0.1  
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
    CaM = Z**n0/(KCa**n0+Z**n0)
    z=np.array([ V0+V1*beta-V2+V3+kf*Y-K*Z,\
                 v1*(MYB15t-MYB15)/(K1P+MYB15t-MYB15)-k1*CaM*MYB15/(K1+MYB15),\
                 ks1-k2*CaM/(Ka1+CaM)*ICE1/(K2+ICE1)*Ka2/(Ka2+CaM)+v2*ICE1P/(K2P+ICE1P)-vd1*ICE1/(Kd1+ICE1),\
                 k2*CaM/(Ka1+CaM)*ICE1/(K2+ICE1)*Ka2/(Ka2+CaM)-v2*ICE1P/(K2P+ICE1P)-vd2*ICE1P/(Kd2+ICE1P)*CaM/(Ka3+CaM),\
                 vs1*CaM*ICE1**n/(KI1**n+ICE1**n)*KI2**m/(KI2**m+MYB15**m)*KI3**r/(KI3**r+ZAT12**r)-vd3*MCBF3/(Kd3+MCBF3),\
                 ks2*MCBF3-k3*CaM*CBF3/(K3+CBF3)+v3*CBF3P/(K3P+CBF3P)-vd4*CBF3/(Kd4+CBF3),\
                 k3*CaM*CBF3/(K3+CBF3)-v3*CBF3P/(K3P+CBF3P)-vd5*CBF3P/(Kd5+CBF3P),\
                 vs2*CBF3**s/(KI4**s+CBF3**s)-vd6*MZAT12/(Kd6+MZAT12),\
                 ks3*MZAT12-vd7*ZAT12/(Kd7+ZAT12),\
                 V2-V3-kf*Y,\
                 vs3*(1-1/(1+(CBF3/KI5)**u+(ZAT12/KI6)**v))-vd8*MCOR15A/(Kd8+MCOR15A),\
                 ks4*MCOR15A-vd9*COR15A/(Kd9+COR15A)])
    return z

tp=2
betaf=0.73
a=6.42
tspan = np.array([0,tp+np.int64(1/a*np.log(5))+1])
h=0.0005
y0 = np.array([0.1,1.96,0.2259,0.4006,0,0.0750,0.0059,0.0088,0.0025,0.6,0.08,0.1])
t=np.arange(tspan[0],tspan[1],h)
y=odeint(fun,y0,t,(betaf,a,))

plt.subplot(132)
plt.plot(t,y[:,0],'k')
#plt.plot(t,y[:,4],'b')
plt.xlim(1.9,tp+1/a*np.log(5))
plt.xlabel('Time (h)',font)
plt.ylabel('[Ca$^{2+}$] ($\mu$M)',font1)
plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.yticks(fontproperties = 'Times New Roman', size = 14)


betaf=0.4
a=6.42
tspan = np.array([0,tp+np.int64(1/a*np.log(5))+1])
h=0.0005
y0 = np.array([0.1,1.96,0.2259,0.4006,0,0.0750,0.0059,0.0088,0.0025,0.6,0.08,0.1])
t=np.arange(tspan[0],tspan[1],h)
y=odeint(fun,y0,t,(betaf,a,))

plt.subplot(133)
plt.plot(t,y[:,0],'k')
#plt.plot(t,y[:,4],'b')
plt.xlim(1.9,tp+1/a*np.log(5))
plt.xlabel('Time (h)',font)
plt.ylabel('[Ca$^{2+}$] ($\mu$M)',font1)
plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.yticks(fontproperties = 'Times New Roman', size = 14)