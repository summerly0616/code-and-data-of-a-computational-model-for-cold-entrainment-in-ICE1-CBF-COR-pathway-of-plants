# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 18:54:56 2021

@author: Huangtin
"""

import numpy as np
import matplotlib.pyplot as plt


def f(t,y1):
#    print(t)
    KCa = 0.14
    V0 = 1*25
    V1 = 7.3*25
#    beta = 0.511
    kf = 1*25
    K = 10*25
    VM2 = 65*25
    KM2 = 1
    VM3 = 500*25
    KM3 = 2
#    KA = 0.9 #the original
    KA = 0.9
    M = 2
    N = 2
    P = 4# the original

   
    tp = 2
    beta = 0.9*np.exp(-8*(t-tp))*(t>tp)
    
    
    Z     = y1[0]
    MYB15 = y1[1]
    ICE1  = y1[2]
    ICE1P = y1[3]
    MCBF3 = y1[4]
    CBF3  = y1[5]
    CBF3P = y1[6]
    MZAT12= y1[7]
    ZAT12 = y1[8]
    Y     = y1[9]
    MCOR15A= y1[10]
    COR15A= y1[11]
    
    V2 = VM2*Z**N/(KM2**N+Z**N)
    V3 = VM3*Y**M/(KM3**M+Y**M)*Z**P/(KA**P+Z**P)
    

    Tc = 2
#    Td =0.05
 
    
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

def f2(t,y2):
#    print(t)
    KCa = 0.14
    V0 = 1*25
    V1 = 7.3*25
#    beta = 0.511
    kf = 1*25
    K = 10*25
    VM2 = 65*25
    KM2 = 1
    VM3 = 500*25
    KM3 = 2
#    KA = 0.9 #the original
    KA = 0.9
    M = 2
    N = 2
    P = 4# the original

   
    tp = 2
    beta = 0.9*np.exp(-0.8*(t-tp))*(t>tp)
    
    
    Z     = y2[0]
    MYB15 = y2[1]
    ICE1  = y2[2]
    ICE1P = y2[3]
    MCBF3 = y2[4]
    CBF3  = y2[5]
    CBF3P = y2[6]
    MZAT12= y2[7]
    ZAT12 = y2[8]
    Y     = y2[9]
    MCOR15A= y2[10]
    COR15A= y2[11]
    
    V2 = VM2*Z**N/(KM2**N+Z**N)
    V3 = VM3*Y**M/(KM3**M+Y**M)*Z**P/(KA**P+Z**P)
    

    Tc = 2
#    Td =0.05
 
    
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

c1=1/2-np.sqrt(3)/6; c2=1/2+np.sqrt(3)/6;
a11=1/4; a12=1/4-np.sqrt(3)/6;
a21=1/4+np.sqrt(3)/6; a22=1/4;
b1=1/2; b2=1/2;

def Gauss2s4(tspan,y0,h):
    t=np.arange(tspan[0],tspan[1]+h,h,dtype='float32')
    n=len(t)-1
    m=len(y0)  
    y1=np.zeros((m,n+1)) 
    y1[:,0]=y0 
    tol=1e-4
    for i in range(0,n,1):
        y1p=y1[:,i]+c1*h*f(t[i],y1[:,i])
        y2p=y1[:,i]+c2*h*f(t[i],y1[:,i])
        y1c=y1[:,i]+h*(a11*f(t[i]+c1*h,y1p)+a12*f(t[i]+c2*h,y2p))  
        y2c=y1[:,i]+h*(a21*f(t[i]+c1*h,y1p)+a22*f(t[i]+c2*h,y2p))  
        while np.linalg.norm(y1p-y1c,ord=2)+np.linalg.norm(y2p-y2c,ord=2)>tol:
              y1p=y1c; y2p=y2c
              y1c=y1[:,i]+h*(a11*f(t[i]+c1*h,y1p)+a12*f(t[i]+c2*h,y2p))  
              y2c=y1[:,i]+h*(a21*f(t[i]+c1*h,y1p)+a22*f(t[i]+c2*h,y2p))
        y1[:,i+1]=y1[:,i]+h*(b1*f(t[i]+c1*h,y1c)+b2*f(t[i]+c2*h,y2c))  
    return t,y1

def Gauss2s4_1(tspan,y0,h):
    t=np.arange(tspan[0],tspan[1]+h,h,dtype='float32')
    n=len(t)-1
    m=len(y0)  
    y2=np.zeros((m,n+1)) 
    y2[:,0]=y0 
    tol=1e-4
    for i in range(0,n,1):
        y1p=y2[:,i]+c1*h*f2(t[i],y2[:,i])
        y2p=y2[:,i]+c2*h*f2(t[i],y2[:,i])
        y1c=y2[:,i]+h*(a11*f2(t[i]+c1*h,y1p)+a12*f(t[i]+c2*h,y2p))  
        y2c=y2[:,i]+h*(a21*f2(t[i]+c1*h,y1p)+a22*f(t[i]+c2*h,y2p))  
        while np.linalg.norm(y1p-y1c,ord=2)+np.linalg.norm(y2p-y2c,ord=2)>tol:
              y1p=y1c; y2p=y2c
              y1c=y2[:,i]+h*(a11*f2(t[i]+c1*h,y1p)+a12*f2(t[i]+c2*h,y2p))  
              y2c=y2[:,i]+h*(a21*f2(t[i]+c1*h,y1p)+a22*f2(t[i]+c2*h,y2p))
        y2[:,i+1]=y2[:,i]+h*(b1*f2(t[i]+c1*h,y1c)+b2*f(t[i]+c2*h,y2c))  
    return t,y2

if __name__ == '__main__':
    y0 = np.array([0.1,1.96,0.2259,0.4006,0,0.0750,0.0059,0.0088,0.0025,0.6,0.01,0.08])
    h=0.0005
    t10=np.arange(0,2+h,h)
    y00=np.ones((len(t10),1))*y0
    y00=y00.T
    tspan = np.array([2,26])
    
    t,y1 = Gauss2s4(tspan, y0 ,h)
    t,y2 = Gauss2s4_1(tspan, y0 ,h)
    t=np.append(t10,t)
    y1=np.append(y00,y1,axis=1)
    y2=np.append(y00,y2,axis=1)


    
    tp = 2
    beta = 0.9*np.exp(-8*(t-tp))*(t>tp)
    beta2 = 0.9*np.exp(-0.8*(t-tp))*(t>tp)
    
    font = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 16,}
    font1 = {'family' : 'Times New Roman',
             'weight' : 'normal',
             'size'   : 14,}

    plt.subplot(221)
    plt.plot(t,y1[0,:],'b')
    plt.plot(t,beta,'r--')
    plt.xlim([1.75,4])
    plt.ylim([-0.1,1.5])
    plt.xlabel('Time(h)',font)
    plt.ylabel(r'Calcium,$\beta$',font1)
    plt.text(2.35, 1.13,'Calcium',color='k',fontdict=font1)
    plt.text(2.56, -0.01,r'$\beta$',color='k',fontsize=14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    
    plt.subplot(222)
    plt.plot(t,y2[0,:],'b')
    plt.plot(t,beta2,'r--')
    plt.xlim([1.75,4])
    plt.ylim([-0.1,1.5])
    plt.xlabel('Time(h)',font)
    plt.ylabel(r'Calcium,$\beta$',font1)
    plt.text(2.8, 1.13,'Calcium',color='k',fontdict=font1)
    plt.text(2.8, 0.4,r'$\beta$',color='k',fontsize=14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    
    plt.subplot(223)
    plt.plot(t,y1[4,:],'k')
    plt.xlim([0,20])
    plt.ylim([-0.1,4])
    plt.xlabel('Time(h)',font)
    plt.ylabel('CBF3 mRNA',font1)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    
    plt.subplot(224)
    plt.plot(t,y2[4,:],'k')
    plt.xlim([0,20])
    plt.ylim([-0.1,4])
    plt.xlabel('Time(h)',font)
    plt.ylabel('CBF3 mRNA',font1)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    
    
    plt.show()
    