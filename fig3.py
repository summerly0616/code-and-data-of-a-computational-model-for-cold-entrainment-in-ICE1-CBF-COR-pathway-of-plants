# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 10:27:15 2021

@author: Huangting
"""

import numpy as np
import matplotlib.pyplot as plt


def f1(t,y):
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
    
    KCa = 0.14
    Tc = 2
    Td = 0.05
    Iin0 = 0.28
    Iin1 = 20
    Iex0 = 2.8
    Iin =(t-Tc>0)*(Tc+Td-t>0)*Iin1

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
    Ka4= 4.5
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
    CaM = Ca**n0/(KCa**n0+Ca**n0)


    z=np.array([ Iin0+Iin-Iex0*Ca,\
                 v1*(MYB15t-MYB15)/(K1P+MYB15t-MYB15)-k1*CaM*MYB15/(K1+MYB15),\
                 ks1-k2*CaM/(Ka1+CaM)*ICE1/(K2+ICE1)*Ka2/(Ka2+CaM)+v2*ICE1P/(K2P+ICE1P)-vd1*ICE1/(Kd1+ICE1)*Ka4/(Ka4+CaM),\
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
    
    KCa = 0.14
    Tc = 2
    Td = 0.05
    Iin0 = 0.28
    Iin1 = 20
    Iex0 = 2.8
    Iin =(t-Tc>0)*(Tc+Td-t>0)*Iin1

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
    ks1= 0.18/5
    ks2= 0.25
    ks3= 2.23
    ks4= 1.2
    vs1= 34  
    vs2= 9.5  
    vs3= 1
    Ka1= 0.35
    Ka2= 0.3
    Ka3= 0.6
    Ka4= 4.5
    KI1= 0.21
    KI2= 0.8  
    KI3= 0.3 
    KI4= 0.28
    KI5= 0.1
    KI6= 0.2
    vd1= 0.1
    vd2= 1.1
    vd3= (t-Tc>0)*10.5+(Tc-t>0)*10.5
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
    CaM = Ca**n0/(KCa**n0+Ca**n0)


    z=np.array([ Iin0+Iin-Iex0*Ca,\
                 v1*(MYB15t-MYB15)/(K1P+MYB15t-MYB15)-k1*CaM*MYB15/(K1+MYB15),\
                 ks1-k2*CaM/(Ka1+CaM)*ICE1/(K2+ICE1)*Ka2/(Ka2+CaM)+v2*ICE1P/(K2P+ICE1P)-vd1*ICE1/(Kd1+ICE1)*Ka4/(Ka4+CaM),\
                 k2*CaM/(Ka1+CaM)*ICE1/(K2+ICE1)*Ka2/(Ka2+CaM)-v2*ICE1P/(K2P+ICE1P)-vd2*ICE1P/(Kd2+ICE1P)*CaM/(Ka3+CaM),\
                 vs1*CaM*ICE1**n/(KI1**n+ICE1**n)*KI2**m/(KI2**m+MYB15**m)*KI3**r/(KI3**r+ZAT12**r)-vd3*MCBF3/(Kd3+MCBF3),\
                 ks2*MCBF3-k3*CaM*CBF3/(K3+CBF3)+v3*CBF3P/(K3P+CBF3P)-vd4*CBF3/(Kd4+CBF3),\
                 k3*CaM*CBF3/(K3+CBF3)-v3*CBF3P/(K3P+CBF3P)-vd5*CBF3P/(Kd5+CBF3P),\
                 vs2*CBF3**s/(KI4**s+CBF3**s)-vd6*MZAT12/(Kd6+MZAT12),\
                 ks3*MZAT12-vd7*ZAT12/(Kd7+ZAT12),\
                 vs3*(1-1/(1+(CBF3/KI5)**u+(ZAT12/KI6)**v))-vd8*MCOR15A/(Kd8+MCOR15A),\
                 ks4*MCOR15A-vd9*COR15A/(Kd9+COR15A)])
    return z

def f3(t,y):
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
    
    KCa = 0.14
    Tc = 2
    Td = 0.05
    Iin0 = 0.28
    Iin1 = 20
    Iex0 = 2.8
    Iin =(t-Tc>0)*(Tc+Td-t>0)*Iin1

    MYB15t = 2

        
    v1 = 0.4  
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
    Ka4= 4.5
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
    CaM = Ca**n0/(KCa**n0+Ca**n0)


    z=np.array([ Iin0+Iin-Iex0*Ca,\
                 v1*(MYB15t-MYB15)/(K1P+MYB15t-MYB15)-k1*CaM*MYB15/(K1+MYB15),\
                 ks1-k2*CaM/(Ka1+CaM)*ICE1/(K2+ICE1)*Ka2/(Ka2+CaM)+v2*ICE1P/(K2P+ICE1P)-vd1*ICE1/(Kd1+ICE1)*Ka4/(Ka4+CaM),\
                 k2*CaM/(Ka1+CaM)*ICE1/(K2+ICE1)*Ka2/(Ka2+CaM)-v2*ICE1P/(K2P+ICE1P)-vd2*ICE1P/(Kd2+ICE1P)*CaM/(Ka3+CaM),\
                 vs1*CaM*ICE1**n/(KI1**n+ICE1**n)*KI2**m/(KI2**m+MYB15**m)*KI3**r/(KI3**r+ZAT12**r)-vd3*MCBF3/(Kd3+MCBF3),\
                 ks2*MCBF3-k3*CaM*CBF3/(K3+CBF3)+v3*CBF3P/(K3P+CBF3P)-vd4*CBF3/(Kd4+CBF3),\
                 k3*CaM*CBF3/(K3+CBF3)-v3*CBF3P/(K3P+CBF3P)-vd5*CBF3P/(Kd5+CBF3P),\
                 vs2*CBF3**s/(KI4**s+CBF3**s)-vd6*MZAT12/(Kd6+MZAT12),\
                 ks3*MZAT12-vd7*ZAT12/(Kd7+ZAT12),\
                 vs3*(1-1/(1+(CBF3/KI5)**u+(ZAT12/KI6)**v))-vd8*MCOR15A/(Kd8+MCOR15A),\
                 ks4*MCOR15A-vd9*COR15A/(Kd9+COR15A)])
    return z


def f4(t,y):
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
    
    KCa = 0.14
    Tc = 2
    Td = 0.05
    Iin0 = 0.28
    Iin1 = 20
    Iex0 = 2.8
    Iin =(t-Tc>0)*(Tc+Td-t>0)*Iin1

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
    vs2= 47.5  
    vs3= 1
    Ka1= 0.35
    Ka2= 0.3
    Ka3= 0.6
    Ka4= 4.5
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
    CaM = Ca**n0/(KCa**n0+Ca**n0)


    z=np.array([ Iin0+Iin-Iex0*Ca,\
                 v1*(MYB15t-MYB15)/(K1P+MYB15t-MYB15)-k1*CaM*MYB15/(K1+MYB15),\
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

def Gauss2s4_1(tspan,y0,h):
    t=np.arange(tspan[0],tspan[1]+h,h,dtype='float32')
    n=len(t)-1
    m=len(y0)  
    y=np.zeros((m,n+1)) 
    y[:,0]=y0 
    tol=1e-4
    for i in range(0,n,1):
        y1p=y[:,i]+c1*h*f1(t[i],y[:,i])
        y2p=y[:,i]+c2*h*f1(t[i],y[:,i])
        y1c=y[:,i]+h*(a11*f1(t[i]+c1*h,y1p)+a12*f1(t[i]+c2*h,y2p))  
        y2c=y[:,i]+h*(a21*f1(t[i]+c1*h,y1p)+a22*f1(t[i]+c2*h,y2p))  
        while np.linalg.norm(y1p-y1c,ord=2)+np.linalg.norm(y2p-y2c,ord=2)>tol:
              y1p=y1c; y2p=y2c
              y1c=y[:,i]+h*(a11*f1(t[i]+c1*h,y1p)+a12*f1(t[i]+c2*h,y2p))  
              y2c=y[:,i]+h*(a21*f1(t[i]+c1*h,y1p)+a22*f1(t[i]+c2*h,y2p))
        y[:,i+1]=y[:,i]+h*(b1*f1(t[i]+c1*h,y1c)+b2*f1(t[i]+c2*h,y2c))  
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


def Gauss2s4_4(tspan,y0,h):
     t=np.arange(tspan[0],tspan[1]+h,h,dtype='float32')
     n=len(t)-1
     m=len(y0)  
     y=np.zeros((m,n+1)) 
     y[:,0]=y0 
     tol=1e-4
     for i in range(0,n,1):
         y1p=y[:,i]+c1*h*f4(t[i],y[:,i])
         y2p=y[:,i]+c2*h*f4(t[i],y[:,i])
         y1c=y[:,i]+h*(a11*f4(t[i]+c1*h,y1p)+a12*f4(t[i]+c2*h,y2p))  
         y2c=y[:,i]+h*(a21*f4(t[i]+c1*h,y1p)+a22*f4(t[i]+c2*h,y2p))  
         while np.linalg.norm(y1p-y1c,ord=2)+np.linalg.norm(y2p-y2c,ord=2)>tol:
               y1p=y1c; y2p=y2c
               y1c=y[:,i]+h*(a11*f4(t[i]+c1*h,y1p)+a12*f4(t[i]+c2*h,y2p))  
               y2c=y[:,i]+h*(a21*f4(t[i]+c1*h,y1p)+a22*f4(t[i]+c2*h,y2p))
         y[:,i+1]=y[:,i]+h*(b1*f4(t[i]+c1*h,y1c)+b2*f4(t[i]+c2*h,y2c))  
     return t,y

if __name__ == '__main__':
    y0 = np.array([0.1,1.92,0.2259,0.4006,0.05,0.0750,0.0059,0.01,0,0.08,0.1])
    h=0.02
    #tspan = np.array([0,25])
    t10=np.arange(0,2+h,h)
    y00=np.ones((len(t10),1))*y0
    y00=y00.T

    tspan = np.array([2,26.5]) 
    t,y1 = Gauss2s4_1(tspan, y0 ,h)

    t,y2 = Gauss2s4_2(tspan, y0 ,h)
    t,y3 = Gauss2s4_3(tspan, y0 ,h)
    t,y4 = Gauss2s4_4(tspan, y0 ,h)

    t=np.append(t10,t)
    y1=np.append(y00,y1,axis=1)
    y2=np.append(y00,y2,axis=1)
    y3=np.append(y00,y3,axis=1)
    y4=np.append(y00,y4,axis=1)
    
    
    font = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 16,}
    font1 = {'family' : 'Times New Roman',
             'weight' : 'normal',
             'size'   : 14,}
    
    
    
    plt.subplot(131)
    plt.plot(t,y1[4,:],'k',label='WT(Sim.)')
    t_CBF_1 = [2,3,5,8,14,26]
    y_CBF_WT1 = [0.05,1.415384615,2.138461538,2.246153846,1.876923077,0.507692308]
    plt.scatter(t_CBF_1,y_CBF_WT1,s=10,color='k',label='WT(Exp.)')
    plt.plot(t,y2[4,:],'b',label='$\it{ice1}$(Sim.)')
    y_CBF_mu1 = [0.05,0.055384615,0.05,0.030769231,0.015384615,0]
    plt.scatter(t_CBF_1,y_CBF_mu1,s=10,color='b',marker='^',label='$\it{ice1}$(Exp.)')
    plt.xlim([0,26.5])
    plt.ylim([-0.1,3])
    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 10})
    plt.xlabel('Time(h)',font)
    plt.ylabel('relative abundance of CBF3 mRNA',font1)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    x1 = np.linspace(0,2,2)
    x2 = np.linspace(2,26.5,2)
    yy1 = 2.85*np.ones(len(x1))
    yy2 = 3*np.ones(len(x1))
    plt.fill_between(x1,yy1,yy2,facecolor='red')
    plt.fill_between(x2,yy1,yy2,facecolor='aliceblue')

    plt.subplot(132)
    plt.plot(t,y1[4,:],'k',label='WT(Sim.)')
    t_CBF_2 = [2,5,8,14,26]
    y_CBF_WT2 = [0.05,3.430327869/1.9,1.549180328/1.9,1.143442623/1.9,0.147540984/1.9]
    plt.scatter(t_CBF_2,y_CBF_WT2,s=10,color='k',label='WT(Exp.)')
    plt.plot(t,y3[4,:],'b',label='$\it{myb15}$(Sim.)')
    y_CBF_mu2 = [0.05,6.786885246/1.2,5.93852459/1.2,1.549180328/1.2,0.295081967/1.2]
    plt.scatter(t_CBF_2,y_CBF_mu2,s=10,color='b',marker='^',label='$\it{myb15}$(Exp.)')
    plt.xlim([0,26.5])
    plt.ylim([-0.1,6])
    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 10})
    plt.xlabel('Time(h)',font)
    plt.ylabel('relative abundance of CBF3 mRNA',font1)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    x1 = np.linspace(0,2,2)
    x2 = np.linspace(2,26.5,2)
    yy1 = 5.7*np.ones(len(x1))
    yy2 = 6*np.ones(len(x1))
    plt.fill_between(x1,yy1,yy2,facecolor='red')
    plt.fill_between(x2,yy1,yy2,facecolor='aliceblue')

    plt.subplot(133)
    plt.plot(t,y1[4,:],'k',label='WT(Sim.)')
    t_CBF_3 = [2,3,4,6,10,26]
    y_CBF_WT3 = [0.05,1.953846154,3.276923077,3.4,1.415384615,0.892307692]
    plt.scatter(t_CBF_3,y_CBF_WT3,s=10,color='k',label='WT(Exp.)')
    plt.plot(t,y4[4,:],'b',label='ZAT12-OE(Sim.)')
    y_CBF_OE = [0.05,0.476923077,1.492307692,1.430769231,0.630769231,0.6]
    plt.scatter(t_CBF_3,y_CBF_OE,s=10,color='b',marker='^',label='ZAT12-OE(Exp.)')
    plt.xlim([0,26.5])
    plt.ylim([-0.1,4])
    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 10})
    
    plt.xlabel('Time(h)',font)
    plt.ylabel('relative abundance of CBF3 mRNA',font1)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    x1 = np.linspace(0,2,2)
    x2 = np.linspace(2,26.5,2)
    yy1 = 3.8*np.ones(len(x1))
    yy2 = 4*np.ones(len(x1))
    plt.fill_between(x1,yy1,yy2,facecolor='red')
    plt.fill_between(x2,yy1,yy2,facecolor='aliceblue')

#    plt.subplot(336)
#    plt.plot(t,y[7,:],'k',label='ZAT12 mRNA')
#    plt.ylim([0,4])
#    plt.legend()
#    t_ZAT12 = [2,3,4,6,10,24]
#    y_ZAT12 = [0.043668122,1.1945701357466063,1.3755656108597285,2.8778280542986425,2.081447963800905,0.9773755656108597]
#    plt.scatter(t_ZAT12,y_ZAT12,color='b')
#    plt.xlabel('time(h)')