# -*- coding: utf-8 -*-
"""
Created on Wed May 19 14:35:02 2021

@author: Huangtin
"""

import numpy as np
import matplotlib.pyplot as plt


def f(t,y):
    KCa = 0.5
    Ca    = y[0]
    MYB15 = y[1]
    ICE1  = y[2]
    ICE1P = y[3]
    MCBF3 = y[4]
    CBF3  = y[5]
    CBF3P = y[6]
    MZAT12= y[7]
    ZAT12 = y[8]
    MCOR15A= y[9]
    COR15A= y[10]
    Tp = 3
    Td =0.05
    Tc1 = Tp/2
    Tc2 = Tp*3/2
    Tc3 = Tp*5/2
    Tc4 = Tp*7/2
    Tc5 = Tp*9/2
    Tc6 = Tp*11/2
    Tc7 = Tp*13/2
    Tc8 = Tp*15/2
    Iin0 = 0.28
    Iin1 = 20
    Iex0 = 2.8
    Iin = (t-Tc1>0)*(Tc1+Td-t>0)*Iin1+(t-Tc2>0)*(Tc2+Td-t>0)*Iin1+(t-Tc3>0)*(Tc3+Td-t>0)*Iin1+(t-Tc4>0)*(Tc4+Td-t>0)*Iin1+(t-Tc5>0)*(Tc5+Td-t>0)*Iin1+(t-Tc6>0)*(Tc6+Td-t>0)*Iin1+(t-Tc7>0)*(Tc7+Td-t>0)*Iin1+(t-Tc8>0)*(Tc8+Td-t>0)*Iin1  
    MYB15t = 2       
    v1 = 4.5  
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
    Ka4= 2.5
    KI1= 0.21
    KI2= 0.8  
    KI3= 0.3 
    KI4= 0.28
    KI5= 0.1
    KI6= 0.2
    vd1= 0.1
    vd2= 1.1
    vd3= ((t-Tc1>0)*(Tp-t>0)+(t-Tc2>0)*(2*Tp-t>0)+(t-Tc3>0)*(3*Tp-t>0)+(t-Tc4>0)*(4*Tp-t>0)+(t-Tc5>0)*(5*Tp-t>0)+(t-Tc6>0)*(6*Tp-t>0)+(t-Tc7>0)*(7*Tp-t>0)+(t-Tc8>0)*(8*Tp-t>0))*0.55+((Tc1-t>0)+(Tc2-t>0)*(t-Tp>0)+(Tc3-t>0)*(t-2*Tp>0)+(Tc4-t>0)*(t-3*Tp>0)+(Tc5-t>0)*(t-4*Tp>0)+(Tc6-t>0)*(t-5*Tp>0)+(Tc7-t>0)*(t-6*Tp>0)+(Tc8-t>0)*(t-7*Tp>0))*5.5
    vd4= 1.2
    vd5= 1.8
    vd6= 0.001  
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
    CaM = Ca**n0/(KCa**n0+Ca**n0)
    z=np.array([ Iin0+Iin-Iex0*Ca,\
                 v1*(MYB15t-MYB15)/(K1P+MYB15t-MYB15)-k1*CaM*MYB15/(K1+MYB15),\
                 ks1-k2*CaM/(Ka1+CaM)*ICE1/(K2+ICE1)*Ka2/(Ka2+CaM)+v2*ICE1P/(K2P+ICE1P)-vd1*ICE1/(Kd1+ICE1)*(Ka4/(Ka4+CaM)*((t-Tc1>0)*(Tp-t>0)+(t-Tc2>0)*(2*Tp-t>0)+(t-Tc3>0)*(3*Tp-t>0)+(t-Tc4>0)*(4*Tp-t>0)+(t-Tc5>0)*(5*Tp-t>0)+(t-Tc6>0)*(6*Tp-t>0)+(t-Tc7>0)*(7*Tp-t>0)+(t-Tc8>0)*(8*Tp-t>0))+1*((Tc1-t>0)+(Tc2-t>0)*(t-Tp>0)+(Tc3-t>0)*(t-2*Tp>0)+(Tc4-t>0)*(t-3*Tp>0)+(Tc5-t>0)*(t-4*Tp>0)+(Tc6-t>0)*(t-5*Tp>0)+(Tc7-t>0)*(t-6*Tp>0)+(Tc8-t>0)*(t-7*Tp>0))),\
                 k2*CaM/(Ka1+CaM)*ICE1/(K2+ICE1)*Ka2/(Ka2+CaM)-v2*ICE1P/(K2P+ICE1P)-vd2*ICE1P/(Kd2+ICE1P)*CaM/(Ka3+CaM),\
                 vs1*CaM*ICE1**n/(KI1**n+ICE1**n)*KI2**m/(KI2**m+MYB15**m)*KI3**r/(KI3**r+ZAT12**r)-vd3*MCBF3/(Kd3+MCBF3),\
                 ks2*MCBF3-k3*CaM*CBF3/(K3+CBF3)+v3*CBF3P/(K3P+CBF3P)-vd4*CBF3/(Kd4+CBF3),\
                 k3*CaM*CBF3/(K3+CBF3)-v3*CBF3P/(K3P+CBF3P)-vd5*CBF3P/(Kd5+CBF3P),\
                 vs2*CBF3**s/(KI4**s+CBF3**s)-vd6*MZAT12/(Kd6+MZAT12),\
                 ks3*MZAT12-vd7*ZAT12/(Kd7+ZAT12),\
                 vs3*(1-1/(1+(CBF3/KI5)**u+(ZAT12/KI6)**v))-vd8*MCOR15A/(Kd8+MCOR15A),\
                 ks4*MCOR15A-vd9*COR15A/(Kd9+COR15A)])
    return z

def f2(t,y):
    KCa = 0.5
    
    
    Ca    = y[0]
    MYB15 = y[1]
    ICE1  = y[2]
    ICE1P = y[3]
    MCBF3 = y[4]
    CBF3  = y[5]
    CBF3P = y[6]
    MZAT12= y[7]
    ZAT12 = y[8]
    MCOR15A= y[9]
    COR15A= y[10]

    Tp = 6
    Td =0.05
    Tc1 = Tp/2
    Tc2 = Tp*3/2
    Tc3 = Tp*5/2
    Tc4 = Tp*7/2
    Tc5 = Tp*9/2
    Tc6 = Tp*11/2
    Tc7 = Tp*13/2
    Tc8 = Tp*15/2
    
    
    
    Iin0 = 0.28
    Iin1 = 20
    Iex0 = 2.8
    Iin = (t-Tc1>0)*(Tc1+Td-t>0)*Iin1+(t-Tc2>0)*(Tc2+Td-t>0)*Iin1+(t-Tc3>0)*(Tc3+Td-t>0)*Iin1+(t-Tc4>0)*(Tc4+Td-t>0)*Iin1+(t-Tc5>0)*(Tc5+Td-t>0)*Iin1+(t-Tc6>0)*(Tc6+Td-t>0)*Iin1+(t-Tc7>0)*(Tc7+Td-t>0)*Iin1+(t-Tc8>0)*(Tc8+Td-t>0)*Iin1    
    MYB15t = 2
    
        
    v1 = 4.5  
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
    Ka4= 2.5
    KI1= 0.21
    KI2= 0.8  
    KI3= 0.3 
    KI4= 0.28
    KI5= 0.1
    KI6= 0.2
    vd1= 0.1
    vd2= 1.1
    vd3= ((t-Tc1>0)*(Tp-t>0)+(t-Tc2>0)*(2*Tp-t>0)+(t-Tc3>0)*(3*Tp-t>0)+(t-Tc4>0)*(4*Tp-t>0)+(t-Tc5>0)*(5*Tp-t>0)+(t-Tc6>0)*(6*Tp-t>0)+(t-Tc7>0)*(7*Tp-t>0)+(t-Tc8>0)*(8*Tp-t>0))*0.55+((Tc1-t>0)+(Tc2-t>0)*(t-Tp>0)+(Tc3-t>0)*(t-2*Tp>0)+(Tc4-t>0)*(t-3*Tp>0)+(Tc5-t>0)*(t-4*Tp>0)+(Tc6-t>0)*(t-5*Tp>0)+(Tc7-t>0)*(t-6*Tp>0)+(Tc8-t>0)*(t-7*Tp>0))*5.5
    vd4= 1.2
    vd5= 1.8
    vd6= 0.001  
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
    CaM = Ca**n0/(KCa**n0+Ca**n0)
    kd = 0
    
    
    z=np.array([ Iin0+Iin-Iex0*Ca,\
                 v1*(MYB15t-MYB15)/(K1P+MYB15t-MYB15)-k1*CaM*MYB15/(K1+MYB15),\
                 ks1-k2*CaM/(Ka1+CaM)*ICE1/(K2+ICE1)*Ka2/(Ka2+CaM)+v2*ICE1P/(K2P+ICE1P)-vd1*ICE1/(Kd1+ICE1)*(Ka4/(Ka4+CaM)*((t-Tc1>0)*(Tp-t>0)+(t-Tc2>0)*(2*Tp-t>0)+(t-Tc3>0)*(3*Tp-t>0)+(t-Tc4>0)*(4*Tp-t>0)+(t-Tc5>0)*(5*Tp-t>0)+(t-Tc6>0)*(6*Tp-t>0)+(t-Tc7>0)*(7*Tp-t>0)+(t-Tc8>0)*(8*Tp-t>0))+1*((Tc1-t>0)+(Tc2-t>0)*(t-Tp>0)+(Tc3-t>0)*(t-2*Tp>0)+(Tc4-t>0)*(t-3*Tp>0)+(Tc5-t>0)*(t-4*Tp>0)+(Tc6-t>0)*(t-5*Tp>0)+(Tc7-t>0)*(t-6*Tp>0)+(Tc8-t>0)*(t-7*Tp>0))),\
                 k2*CaM/(Ka1+CaM)*ICE1/(K2+ICE1)*Ka2/(Ka2+CaM)-v2*ICE1P/(K2P+ICE1P)-vd2*ICE1P/(Kd2+ICE1P)*CaM/(Ka3+CaM),\
                 vs1*CaM*ICE1**n/(KI1**n+ICE1**n)*KI2**m/(KI2**m+MYB15**m)*KI3**r/(KI3**r+ZAT12**r)-vd3*MCBF3/(Kd3+MCBF3)-kd*MCBF3,\
                 ks2*MCBF3-k3*CaM*CBF3/(K3+CBF3)+v3*CBF3P/(K3P+CBF3P)-vd4*CBF3/(Kd4+CBF3),\
                 k3*CaM*CBF3/(K3+CBF3)-v3*CBF3P/(K3P+CBF3P)-vd5*CBF3P/(Kd5+CBF3P),\
                 vs2*CBF3**s/(KI4**s+CBF3**s)-vd6*MZAT12/(Kd6+MZAT12),\
                 ks3*MZAT12-vd7*ZAT12/(Kd7+ZAT12),\
                 vs3*(1-1/(1+(CBF3/KI5)**u+(ZAT12/KI6)**v))-vd8*MCOR15A/(Kd8+MCOR15A),\
                 ks4*MCOR15A-vd9*COR15A/(Kd9+COR15A)])
    return z



def f3(t,y):
    KCa = 0.5
    
    
    Ca    = y[0]
    MYB15 = y[1]
    ICE1  = y[2]
    ICE1P = y[3]
    MCBF3 = y[4]
    CBF3  = y[5]
    CBF3P = y[6]
    MZAT12= y[7]
    ZAT12 = y[8]
    MCOR15A= y[9]
    COR15A= y[10]

    Tp = 12
    Td =0.05
    Tc1 = Tp/2
    Tc2 = Tp*3/2
    Tc3 = Tp*5/2
    Tc4 = Tp*7/2
    Tc5 = Tp*9/2
    Tc6 = Tp*11/2
    Tc7 = Tp*13/2
    Tc8 = Tp*15/2
    
    
    Iin0 = 0.28
    Iin1 = 20
    Iex0 = 2.8
    Iin = (t-Tc1>0)*(Tc1+Td-t>0)*Iin1+(t-Tc2>0)*(Tc2+Td-t>0)*Iin1+(t-Tc3>0)*(Tc3+Td-t>0)*Iin1+(t-Tc4>0)*(Tc4+Td-t>0)*Iin1+(t-Tc5>0)*(Tc5+Td-t>0)*Iin1+(t-Tc6>0)*(Tc6+Td-t>0)*Iin1+(t-Tc7>0)*(Tc7+Td-t>0)*Iin1+(t-Tc8>0)*(Tc8+Td-t>0)*Iin1
 
    
    MYB15t = 2
    
        
    v1 = 4.5  
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
    Ka4= 2.5
    KI1= 0.21
    KI2= 0.8  
    KI3= 0.3 
    KI4= 0.28
    KI5= 0.1
    KI6= 0.2
    vd1= 0.1
    vd2= 1.1
    vd3= ((t-Tc1>0)*(Tp-t>0)+(t-Tc2>0)*(2*Tp-t>0)+(t-Tc3>0)*(3*Tp-t>0)+(t-Tc4>0)*(4*Tp-t>0)+(t-Tc5>0)*(5*Tp-t>0)+(t-Tc6>0)*(6*Tp-t>0)+(t-Tc7>0)*(7*Tp-t>0)+(t-Tc8>0)*(8*Tp-t>0))*0.55+((Tc1-t>0)+(Tc2-t>0)*(t-Tp>0)+(Tc3-t>0)*(t-2*Tp>0)+(Tc4-t>0)*(t-3*Tp>0)+(Tc5-t>0)*(t-4*Tp>0)+(Tc6-t>0)*(t-5*Tp>0)+(Tc7-t>0)*(t-6*Tp>0)+(Tc8-t>0)*(t-7*Tp>0))*5.5
    vd4= 1.2
    vd5= 1.8
    vd6= 0.001  
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
    CaM = Ca**n0/(KCa**n0+Ca**n0)

    
    
    z=np.array([ Iin0+Iin-Iex0*Ca,\
                 v1*(MYB15t-MYB15)/(K1P+MYB15t-MYB15)-k1*CaM*MYB15/(K1+MYB15),\
                 ks1-k2*CaM/(Ka1+CaM)*ICE1/(K2+ICE1)*Ka2/(Ka2+CaM)+v2*ICE1P/(K2P+ICE1P)-vd1*ICE1/(Kd1+ICE1)*(Ka4/(Ka4+CaM)*((t-Tc1>0)*(Tp-t>0)+(t-Tc2>0)*(2*Tp-t>0)+(t-Tc3>0)*(3*Tp-t>0)+(t-Tc4>0)*(4*Tp-t>0)+(t-Tc5>0)*(5*Tp-t>0)+(t-Tc6>0)*(6*Tp-t>0)+(t-Tc7>0)*(7*Tp-t>0)+(t-Tc8>0)*(8*Tp-t>0))+1*((Tc1-t>0)+(Tc2-t>0)*(t-Tp>0)+(Tc3-t>0)*(t-2*Tp>0)+(Tc4-t>0)*(t-3*Tp>0)+(Tc5-t>0)*(t-4*Tp>0)+(Tc6-t>0)*(t-5*Tp>0)+(Tc7-t>0)*(t-6*Tp>0)+(Tc8-t>0)*(t-7*Tp>0))),\
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

def Gauss2s4_2(tspan,y0,h):
    t=np.arange(tspan[0],tspan[1]+h,h,dtype='float32')
    n=len(t)-1
    m=len(y0)  
    y=np.zeros((m,n+1)) 
    y[:,0]=y0 
    tol=1e-4
    for i in range(0,n,1):
        y1p=y[:,i]+c1*h*f2(t[i],y[:,i])
        y2p=y[:,i]+c2*h*f2(t[i],y[:,i])
        y1c=y[:,i]+h*(a11*f2(t[i]+c1*h,y1p)+a12*f2(t[i]+c2*h,y2p))  
        y2c=y[:,i]+h*(a21*f2(t[i]+c1*h,y1p)+a22*f2(t[i]+c2*h,y2p))  
        while np.linalg.norm(y1p-y1c,ord=2)+np.linalg.norm(y2p-y2c,ord=2)>tol:
              y1p=y1c; y2p=y2c
              y1c=y[:,i]+h*(a11*f2(t[i]+c1*h,y1p)+a12*f2(t[i]+c2*h,y2p))  
              y2c=y[:,i]+h*(a21*f2(t[i]+c1*h,y1p)+a22*f2(t[i]+c2*h,y2p))
        y[:,i+1]=y[:,i]+h*(b1*f2(t[i]+c1*h,y1c)+b2*f2(t[i]+c2*h,y2c))  
    return t,y

def Gauss2s4_3(tspan,y0,h):
    t=np.arange(tspan[0],tspan[1]+h,h,dtype='float32')
    n=len(t)-1
    m=len(y0)  
    y=np.zeros((m,n+1)) 
    y[:,0]=y0 
    tol=1e-4
    for i in range(0,n,1):
        y1p=y[:,i]+c1*h*f3(t[i],y[:,i])
        y2p=y[:,i]+c2*h*f3(t[i],y[:,i])
        y1c=y[:,i]+h*(a11*f3(t[i]+c1*h,y1p)+a12*f3(t[i]+c2*h,y2p))  
        y2c=y[:,i]+h*(a21*f3(t[i]+c1*h,y1p)+a22*f3(t[i]+c2*h,y2p))  
        while np.linalg.norm(y1p-y1c,ord=2)+np.linalg.norm(y2p-y2c,ord=2)>tol:
              y1p=y1c; y2p=y2c
              y1c=y[:,i]+h*(a11*f3(t[i]+c1*h,y1p)+a12*f3(t[i]+c2*h,y2p))  
              y2c=y[:,i]+h*(a21*f3(t[i]+c1*h,y1p)+a22*f3(t[i]+c2*h,y2p))
        y[:,i+1]=y[:,i]+h*(b1*f3(t[i]+c1*h,y1c)+b2*f3(t[i]+c2*h,y2c))  
    return t,y



if __name__ == '__main__':
    y0 = np.array([0.1,1.92,0.2259,0.2981,0.01,0.0750,0.0059,0.05,0.05,0,0.01])
    y02 = np.array([0.1,1.5,0.2,0.2981,0.01,0.0750,0.0059,0.05,0.05,0,0.01])
    h=0.005
    
    t10=np.arange(0,1.5+h,h)
    y10=np.ones((len(t10),1))*y0
    y10=y10.T
    tspan1 = np.array([1.5,18.5])
    t1,y1 = Gauss2s4(tspan1, y0 ,h)
    t1=np.append(t10,t1)
    y1=np.append(y10,y1,axis=1)
     
    t20=np.arange(0,3+h,h)
    y20=np.ones((len(t20),1))*y0
    y20=y20.T
    tspan2 = np.array([3,36.5]) 
    t2,y2 = Gauss2s4_2(tspan2, y0 ,h)
    t2=np.append(t20,t2)
    y2=np.append(y20,y2,axis=1)    
#    
    t30=np.arange(0,6+h,h)
    y30=np.ones((len(t30),1))*y02
    y30=y30.T
    tspan3 = np.array([6,72.5])
    t3,y3 = Gauss2s4_3(tspan3, y02 ,h)
    t3=np.append(t30,t3)
    y3=np.append(y30,y3,axis=1)
    
    font = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 16,}
    font1 = {'family' : 'Times New Roman',
             'weight' : 'normal',
             'size'   : 14,}

    
    plt.subplot(231)    
    plt.plot(t1,y1[4,:],'k',label='CBF3 mRNA')
    t_CBF = [1.5,3,4.5,6,7.5,9,10.5,12]
    y_CBF = [0.01,0.4159,0.042438776,0.44985102,0.048097279,0.38194898,0.01697551,0.169755102]
    plt.scatter(t_CBF,y_CBF,s=12,color='r',label='CBF3 mRNA(Exp.)')
    plt.xlim([0,18.5])
    plt.ylim([0,1.5])
    plt.xlabel('Time(h)',font)
    plt.ylabel('CBF3 mRNA',font1)
    plt.title('warm/cold cycle = 1.5/1.5',font1)
#    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 10})
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    xx = np.linspace(0,1.5,2)
    xxx = np.linspace(18,18.5,2)
    yy1 = 1.44*np.ones(len(xx))
    yy2 = 1.5*np.ones(len(xx))
    yyy1 = 1.44*np.ones(len(xxx))
    yyy2 = 1.5*np.ones(len(xxx))
    for i in range(6):
        xx1=xx+3.0*i
        plt.fill_between(xx1,yy1,yy2,facecolor='red')
        plt.fill_between(xx1+1.5,yy1,yy2,facecolor='aliceblue')
    plt.fill_between(xxx,yyy1,yyy2,facecolor='red')
    
    
    plt.subplot(232)    
    plt.plot(t2,y2[4,:],'k',label='CBF3 mRNA')
    plt.xlim([0,36.5])
    plt.ylim([0,1.5])
    plt.xlabel('Time(h)',font)
    plt.ylabel('CBF3 mRNA',font1)
    plt.title('warm/cold cycle = 3/3',font1)
#    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 10})
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    x2 = np.linspace(0,3,2)
    x3 = np.linspace(36,36.5,2)
    yyy3 = 1.44*np.ones(len(x3))
    yyy4 = 1.5*np.ones(len(x3))
    plt.fill_between(x3,yyy3,yyy4,facecolor='red')
    for i in range(6):
        xx2=x2+6.0*i
        plt.fill_between(xx2,yy1,yy2,facecolor='red')
        plt.fill_between(xx2+3,yy1,yy2,facecolor='aliceblue')
    
    
    
    plt.subplot(233)    
    plt.plot(t3,y3[4,:],'k',label='CBF3 mRNA')
    plt.xlim([0,72.5])
    plt.ylim([0,1.5])
    plt.xlabel('Time(h)',font)
    plt.ylabel('CBF3 mRNA',font1)
    plt.title('warm/cold cycle = 6/6',font1)
#    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 10})
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    x3 = np.linspace(0,6,2)
    for i in range(6):
        xx3=x3+12.0*i
        plt.fill_between(xx3,yy1,yy2,facecolor='red')
        plt.fill_between(xx3+6,yy1,yy2,facecolor='aliceblue')
    x4 = np.linspace(72,72.5,2)
    yyy5 = 1.44*np.ones(len(x4))
    yyy6 = 1.5*np.ones(len(x4))
    plt.fill_between(x4,yyy5,yyy6,facecolor='red')

    plt.subplot(234)    
    plt.plot(t1,y1[9,:],'k',label='COR15A mRNA')
    t_COR = [1.5,3,4.5,6,7.5,9,10.5,12]
    y_COR = [0,0.2,0.379136063,0.789598263,1.108685406,1.100711817,1.046158867,1.287966443]
    plt.scatter(t_COR,y_COR,s=12,color='r',label='COR15A mRNA(Exp.)')
    plt.xlim([0,18.5])
    plt.ylim([0,2.5])
    plt.xlabel('Time(h)',font)
    plt.ylabel('COR15A mRNA',font1)
#    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 10})
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    xx2 = np.linspace(0,1.5,2)
    xxx2 = np.linspace(18,18.5,2)
    yyy1 = 2.4*np.ones(len(xx2))
    yyy2 = 2.5*np.ones(len(xx2))
    yyyy1 = 2.4*np.ones(len(xxx2))
    yyyy2 = 2.5*np.ones(len(xxx2))
    for i in range(6):
        xxx1=xx2+3.0*i
        plt.fill_between(xxx1,yyy1,yyy2,facecolor='red')
        plt.fill_between(xxx1+1.5,yyy1,yyy2,facecolor='aliceblue')
    plt.fill_between(xxx2,yyyy1,yyyy2,facecolor='red')
    
    
    plt.subplot(235)    
    plt.plot(t2,y2[9,:],'k',label='COR15A mRNA')
    plt.xlim([0,36.5])
    plt.ylim([0,2.5])
    plt.xlabel('Time(h)',font)
    plt.ylabel('COR15A mRNA',font1)
#    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 10})
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    x5 = np.linspace(0,3,2)
    for i in range(6):
        xx5=x5+6.0*i
        plt.fill_between(xx5,yyy1,yyy2,facecolor='red')
        plt.fill_between(xx5+3,yyy1,yyy2,facecolor='aliceblue')
    x3 = np.linspace(36,36.5,2)
    yyy3 = 2.4*np.ones(len(x3))
    yyy4 = 2.5*np.ones(len(x3))
    plt.fill_between(x3,yyy3,yyy4,facecolor='red')
    
    plt.subplot(236)    
    plt.plot(t3,y3[9,:],'k',label='COR15A mRNA')
    plt.xlim([0,72.5])
    plt.ylim([0,2.5])
    plt.xlabel('Time(h)',font)
    plt.ylabel('COR15A mRNA',font1)
#    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 10})
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    x6 = np.linspace(0,6,2)
    for i in range(6):
        xx6=x6+12.0*i
        plt.fill_between(xx6,yyy1,yyy2,facecolor='red')
        plt.fill_between(xx6+6,yyy1,yyy2,facecolor='aliceblue')
    x4 = np.linspace(72,72.5,2)
    yyy5 = 2.4*np.ones(len(x4))
    yyy6 = 2.5*np.ones(len(x4))
    plt.fill_between(x4,yyy5,yyy6,facecolor='red')
    
    
    
    plt.show()

  
