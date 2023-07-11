# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 17:57:16 2021

@author: Huangtin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def f(t,y):
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
    Ca = 0.1
    CaM = Ca**n0/(KCa**n0+Ca**n0)
#    kd = 0.05

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

c1=1/2-np.sqrt(3)/6; c2=1/2+np.sqrt(3)/6;
a11=1/4; a12=1/4-np.sqrt(3)/6;
a21=1/4+np.sqrt(3)/6; a22=1/4;
b1=1/2; b2=1/2;
def Gauss2s4(tspan,y0,h):
    t=np.arange(tspan[0],tspan[1]+h,h,dtype='float32')
    n=len(t)-1
    m=len(y0)  
    y=np.zeros((m,n+1)) 
    y[:,0]=y0 
    tol=1e-4
    for i in range(0,n,1):
        y1p=y[:,i]+c1*h*f(t[i],y[:,i])
        y2p=y[:,i]+c2*h*f(t[i],y[:,i])
        y1c=y[:,i]+h*(a11*f(t[i]+c1*h,y1p)+a12*f(t[i]+c2*h,y2p))  
        y2c=y[:,i]+h*(a21*f(t[i]+c1*h,y1p)+a22*f(t[i]+c2*h,y2p))  
        while np.linalg.norm(y1p-y1c,ord=2)+np.linalg.norm(y2p-y2c,ord=2)>tol:
              y1p=y1c; y2p=y2c
              y1c=y[:,i]+h*(a11*f(t[i]+c1*h,y1p)+a12*f(t[i]+c2*h,y2p))  
              y2c=y[:,i]+h*(a21*f(t[i]+c1*h,y1p)+a22*f(t[i]+c2*h,y2p))
        y[:,i+1]=y[:,i]+h*(b1*f(t[i]+c1*h,y1c)+b2*f(t[i]+c2*h,y2c))  
    return t,y


if __name__ == '__main__':
    
 #coefficient of funCa  
    
    tspan = np.array([0,300])
    h=0.005
    t=np.arange(tspan[0],tspan[1],h)
    y0 = np.array([1.88,0.2259,0.44,0.34,0.0750,0.0059,0.51,0.01,0.08,0.1])
    y00 = np.array([1.88,0.2259,0.44,0.91,0.0750,0.0059,0.3,0.01,0.08,0.1])
    y000 = np.array([1.88,0.2259,0.44,0.06,0.0750,0.0059,0.56,0.01,0.08,0.1])
    t,y = Gauss2s4(tspan, y0 ,h) 
    t1,y1 = Gauss2s4(tspan, y00 ,h) 
    t2,y2 = Gauss2s4(tspan, y000 ,h) 
    Ca = [0.1 for _ in t]   
    KCa = 0.14
    n0 = 4
    CaM = np.power(Ca,n0)/(np.power(KCa,n0)+np.power(Ca,n0))
        
    font = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 16,}
    font1 = {'family' : 'Times New Roman',
             'weight' : 'normal',
             'size'   : 14,}
    
    plt.subplot(221)
    plt.plot(t,Ca,'b',label='Calcium')
    plt.plot(t,y[3,:],'k',label='CBF3 mRNA')
    plt.plot(t,y[6,:],'r',label='ZAT12 mRNA')
    plt.xlim([100,300])
    plt.ylim([0,3])
    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 10})
    plt.xlabel('Time(h)',font)
    plt.ylabel('CBF3 and ZAT12 mRNA',font1)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    
    plt.subplot(222)
    plt.plot(t,Ca,'b',label='Calcium')
    plt.plot(t,y[4,:],'k',label='CBF3 protein')
    plt.plot(t,y[7,:],'r',label='ZAT12 protein')
    plt.xlim([100,300])
    plt.ylim([0,3])
    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 10})
    plt.xlabel('Time(h)',font)
    plt.ylabel('CBF3 and ZAT12 protein',font1)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    
    plt.subplot(223)
    plt.plot(y[3,20000:],y[6,20000:],'b')
    plt.plot(y1[3,100:20000],y1[6,100:20000],'b')
    plt.plot(y2[3,100:2000],y2[6,100:2000],'r')
    plt.plot(y2[3,20000:],y2[6,20000:],'r')
#    plt.xlim([0,1])
#    plt.ylim([0,3])
    plt.xlabel('CBF mRNA',font)
    plt.ylabel('ZAT12 mRNA',font1)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    
    plt.subplot(224)
    plt.plot(t,Ca,'b',label='Calcium')
    plt.plot(t,CaM,'r',label='CaM')
    plt.plot(t,y[1,:],'g',label='ICE1')
    plt.xlim([100,300])
    plt.ylim([0,3])
    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 10})
    plt.xlabel('Time(h)',font)
    plt.ylabel('Ca,CaM,ICE1',font1)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    
#    peak_id,peak_property = find_peaks(y[4,400000:-1], height=0, distance=20) # height: equilibrium level; distance
#    peak_time = t[peak_id]
#    peak_height = peak_property['peak_heights']
# 
##print(peak_time,peak_height)
#    diff=np.diff(peak_time)
#    period=max(abs(diff))
#    maxx=max(y[4,400000:-1])
#    print(period)
    