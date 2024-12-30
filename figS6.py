# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 11:01:59 2021

@author: Huangtin
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
    
    KCa = 0.14
    Tc = 2
    Td = 0.05
    Iin0 = 0.28
    Iin1 = 20
    Iex0 = 2.8
    Iin =(t-Tc>0)*(Tc+Td-t>0)*Iin1

    MYB15t = 2

        
    k8 = 4.92  #
    k9 = 15
    k11= 2
    v4 = 4  #
    v5 = 1.2
    v7 = 2
    K7 = 0.5 #
    K8 = 0.5 #
    K9 = 0.45
    K10= 0.8
    K13= 0.6
    K14= 0.58
    vs1= 0.18
#    vs1= 0.18/4 # ice1 mutant
#    vs2= 18
    vs2= 11  #
    vs3= 9.5  #
    ks1= 0.25
    ks2= 2.23
    Ka1= 0.6
#    Ka2= 1.45  #
    Ka2= 0.21
    Ka3= 0.28  #
    KI1= 0.3
    KI1a= 0.35
    KI2= 0.8  #
    KI3= 0.3 #
    vd1= 0.1
    vd2= 1.1
#    vd3= (t-Tc>0)*0.55+(Tc-t>0)*5.5
    vd3= (t-Tc>0)*0.55+(Tc-t>0)*5.5
    vd4= 1.2
    vd5= 1.8
    vd6= 0.1  #
    vd7= 1.3
    Kd1= 0.2
    Kd2= 0.4
    Kd3= 0.6  #
    Kd4= 0.1
    Kd5= 0.25
    Kd6= 0.01
    Kd7= 1.25
    n  = 2
    m  = 2
    r  = 2
    s  = 2
    n0 = 4
    CaM = Ca**n0/(KCa**n0+Ca**n0)
    kd3 = 0.05

    z=np.array([ Iin0+Iin-Iex0*Ca,\
                 v4*(MYB15t-MYB15)/(K7+MYB15t-MYB15)-k8*CaM*MYB15/(K8+MYB15),\
                 vs1-k9*CaM/(KI1a+CaM)*ICE1/(K9+ICE1)*KI1/(KI1+CaM)+v5*ICE1P/(K10+ICE1P)-vd1*ICE1/(Kd1+ICE1),\
                 k9*CaM/(KI1a+CaM)*ICE1/(K9+ICE1)*KI1/(KI1+CaM)-v5*ICE1P/(K10+ICE1P)-vd2*ICE1P/(Kd2+ICE1P)*CaM/(Ka1+CaM),\
                 vs2*ICE1P**n/(Ka2**n+ICE1P**n)*KI2**m/(KI2**m+MYB15**m)*KI3**r/(KI3**r+ZAT12**r)-vd3*MCBF3/(Kd3+MCBF3)-kd3*MCBF3,\
                 ks1*MCBF3-k11*CaM*CBF3/(K13+CBF3)+v7*CBF3P/(K14+CBF3P)-vd4*CBF3/(Kd4+CBF3),\
                 k11*CaM*CBF3/(K13+CBF3)-v7*CBF3P/(K14+CBF3P)-vd5*CBF3P/(Kd5+CBF3P),\
                 vs3*CBF3**s/(Ka3**s+CBF3**s)-vd6*MZAT12/(Kd6+MZAT12),\
                 ks2*MZAT12-vd7*ZAT12/(Kd7+ZAT12)])
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
    
    KCa = 0.14
    Tc = 2
    Td = 0.05
    Iin0 = 0.28
    Iin1 = 20
    Iex0 = 2.8
    Iin =(t-Tc>0)*(Tc+Td-t>0)*Iin1

    MYB15t = 2

        
    k8 = 4.92  #
    k9 = 15
    k11= 2
    v4 = 0.4  #
    v5 = 1.2
    v7 = 2
    K7 = 0.5 #
    K8 = 0.5 #
    K9 = 0.45
    K10= 0.8
    K13= 0.6
    K14= 0.58
#    vs1= 0.18
    vs1= 0.18 
#    vs2= 18
    vs2= 11  #
    vs3= 9.5  #
    ks1= 0.25
    ks2= 2.23
    Ka1= 0.6
#    Ka2= 1.45  #
    Ka2= 0.21
    Ka3= 0.28  #
    KI1= 0.3
    KI1a= 0.35
    KI2= 0.8  #
    KI3= 0.3 #
    vd1= 0.1
    vd2= 1.1
#    vd3= (t-Tc>0)*0.55+(Tc-t>0)*5.5
    vd3= (t-Tc>0)*0.55+(Tc-t>0)*5.5
    vd4= 1.2
    vd5= 1.8
    vd6= 0.1  #
    vd7= 1.3
    Kd1= 0.2
    Kd2= 0.4
    Kd3= 0.6  #
    Kd4= 0.1
    Kd5= 0.25
    Kd6= 0.01
    Kd7= 1.25
    n  = 2
    m  = 2
    r  = 2
    s  = 2
    n0 = 4
    CaM = Ca**n0/(KCa**n0+Ca**n0)
    kd3 = 0.05

    z=np.array([ Iin0+Iin-Iex0*Ca,\
                 v4*(MYB15t-MYB15)/(K7+MYB15t-MYB15)-k8*CaM*MYB15/(K8+MYB15),\
                 vs1-k9*CaM/(KI1a+CaM)*ICE1/(K9+ICE1)*KI1/(KI1+CaM)+v5*ICE1P/(K10+ICE1P)-vd1*ICE1/(Kd1+ICE1),\
                 k9*CaM/(KI1a+CaM)*ICE1/(K9+ICE1)*KI1/(KI1+CaM)-v5*ICE1P/(K10+ICE1P)-vd2*ICE1P/(Kd2+ICE1P)*CaM/(Ka1+CaM),\
                 vs2*ICE1P**n/(Ka2**n+ICE1P**n)*KI2**m/(KI2**m+MYB15**m)*KI3**r/(KI3**r+ZAT12**r)-vd3*MCBF3/(Kd3+MCBF3)-kd3*MCBF3,\
                 ks1*MCBF3-k11*CaM*CBF3/(K13+CBF3)+v7*CBF3P/(K14+CBF3P)-vd4*CBF3/(Kd4+CBF3),\
                 k11*CaM*CBF3/(K13+CBF3)-v7*CBF3P/(K14+CBF3P)-vd5*CBF3P/(Kd5+CBF3P),\
                 vs3*CBF3**s/(Ka3**s+CBF3**s)-vd6*MZAT12/(Kd6+MZAT12),\
                 ks2*MZAT12-vd7*ZAT12/(Kd7+ZAT12)])
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
    
    KCa = 0.14
    Tc = 2
    Td = 0.05
    Iin0 = 0.28
    Iin1 = 20
    Iex0 = 2.8
    Iin =(t-Tc>0)*(Tc+Td-t>0)*Iin1

    MYB15t = 2

        
    k8 = 4.92  #
    k9 = 15
    k11= 2
    v4 = 0.4  #
    v5 = 1.2
    v7 = 2
    K7 = 0.5 #
    K8 = 0.5 #
    K9 = 0.45
    K10= 0.8
    K13= 0.6
    K14= 0.58
#    vs1= 0.18
    vs1= 0.18 
#    vs2= 18
    vs2= 11  #
    vs3= 2.2  #
    ks1= 0.25
    ks2= 2.23
    Ka1= 0.6
#    Ka2= 1.45  #
    Ka2= 0.21
    Ka3= 0.28  #
    KI1= 0.3
    KI1a= 0.35
    KI2= 0.8  #
    KI3= 0.3 #
    vd1= 0.1
    vd2= 1.1
#    vd3= (t-Tc>0)*0.55+(Tc-t>0)*5.5
    vd3= (t-Tc>0)*0.55+(Tc-t>0)*5.5
    vd4= 1.2
    vd5= 1.8
    vd6= 0.2  #
    vd7= 1.3
    Kd1= 0.2
    Kd2= 0.4
    Kd3= 0.6  #
    Kd4= 0.1
    Kd5= 0.25
    Kd6= 0.01
    Kd7= 1.25
    n  = 2
    m  = 2
    r  = 2
    s  = 2
    n0 = 4
    CaM = Ca**n0/(KCa**n0+Ca**n0)
    kd3 = 0.15

    z=np.array([ Iin0+Iin-Iex0*Ca,\
                 v4*(MYB15t-MYB15)/(K7+MYB15t-MYB15)-k8*CaM*MYB15/(K8+MYB15),\
                 vs1-k9*CaM/(KI1a+CaM)*ICE1/(K9+ICE1)*KI1/(KI1+CaM)+v5*ICE1P/(K10+ICE1P)-vd1*ICE1/(Kd1+ICE1),\
                 k9*CaM/(KI1a+CaM)*ICE1/(K9+ICE1)*KI1/(KI1+CaM)-v5*ICE1P/(K10+ICE1P)-vd2*ICE1P/(Kd2+ICE1P)*CaM/(Ka1+CaM),\
                 vs2*ICE1P**n/(Ka2**n+ICE1P**n)*KI2**m/(KI2**m+MYB15**m)*KI3**r/(KI3**r+ZAT12**r)-vd3*MCBF3/(Kd3+MCBF3)-kd3*MCBF3,\
                 ks1*MCBF3-k11*CaM*CBF3/(K13+CBF3)+v7*CBF3P/(K14+CBF3P)-vd4*CBF3/(Kd4+CBF3),\
                 k11*CaM*CBF3/(K13+CBF3)-v7*CBF3P/(K14+CBF3P)-vd5*CBF3P/(Kd5+CBF3P),\
                 vs3*CBF3**s/(Ka3**s+CBF3**s)-vd6*MZAT12/(Kd6+MZAT12),\
                 ks2*MZAT12-vd7*ZAT12/(Kd7+ZAT12)])
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
              y1c=y[:,i]+h*(a11*f3(t[i]+c1*h,y1p)+a12*f2(t[i]+c2*h,y2p))  
              y2c=y[:,i]+h*(a21*f3(t[i]+c1*h,y1p)+a22*f2(t[i]+c2*h,y2p))
        y[:,i+1]=y[:,i]+h*(b1*f3(t[i]+c1*h,y1c)+b2*f3(t[i]+c2*h,y2c))  
    return t,y




if __name__ == '__main__':
    y0 = np.array([0.1,1.88,0.2259,0.4006,0.05,0,0.00125,0.01,0])
    y0bar = np.array([0.1,0.21,0.2259,0.4006,0.05,0,0.00125,0.01,0])
    h=0.02
    #tspan = np.array([0,25])
    t10=np.arange(0,2+h,h)
    y00=np.ones((len(t10),1))*y0
    y00=y00.T
    y00bar=np.ones((len(t10),1))*y0bar
    y00bar=y00bar.T
    tspan = np.array([2,26.5]) 
    t,y1 = Gauss2s4_1(tspan, y0 ,h)
    #print(y1.shape)
    t,y2 = Gauss2s4_2(tspan, y0bar ,h)
    t,y3 = Gauss2s4_3(tspan, y0bar ,h)
   
#print(y[:,-5])
#plotting time courses
    t=np.append(t10,t)
    y1=np.append(y00,y1,axis=1)
    y2=np.append(y00bar,y2,axis=1)
    y3=np.append(y00bar,y3,axis=1)
    
    
    
    
    font = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 16,}
    font1 = {'family' : 'Times New Roman',
             'weight' : 'normal',
             'size'   : 14,}
    
    
    
    plt.subplot(221)
    plt.plot(t,y1[1,:],'k',label='WT')
    plt.plot(t,y2[1,:],'b',label='$\it{myb15}$')
    
    plt.xlim([0,24])
#    plt.ylim([0,1.2])
    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 10})
    plt.xlabel('Time(h)',font)
    plt.ylabel('MYB15',font1)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)


    plt.subplot(222)
    plt.plot(t,y1[5,:],'k',label='WT')
    plt.plot(t,y2[5,:],'b',label='$\it{myb15}$')
   
    plt.xlim([0,24])
#    plt.ylim([-0.1,4])
    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 10})
    
    plt.xlabel('Time(h)',font)
    plt.ylabel('CBF3',font1)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    
    plt.subplot(223)
    plt.plot(t,y1[6,:],'k',label='WT')
    plt.plot(t,y2[6,:],'b',label='$\it{myb15}$')
    
    plt.xlim([0,24])
#    plt.ylim([-0.1,4])
    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 10})
    
    plt.xlabel('Time(h)',font)
    plt.ylabel('CBF3P',font1)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    
    plt.subplot(224)
    plt.plot(t,y1[7,:],'k',label='WT')
    plt.plot(t,y3[7,:],'b',label='$\it{myb15}$')
    plt.xlim([0,24])
#    plt.ylim([-0.1,4])
    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 10})
    
    plt.xlabel('Time(h)',font)
    plt.ylabel('ZAT12 mRNA',font1)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)