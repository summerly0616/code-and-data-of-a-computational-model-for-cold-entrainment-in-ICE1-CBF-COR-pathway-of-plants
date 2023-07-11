# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 09:36:36 2021

@author: Huangtin
"""

import numpy as np
import matplotlib.pyplot as plt


def f(t,y):
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
    Ka4= 0.5
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


def f1(t,y1):
    Ca    = y1[0]
    MYB15 = y1[1]
    ICE1  = y1[2]
    ICE1P = y1[3]
    MCBF3 = y1[4]
    CBF3  = y1[5]
    CBF3P = y1[6]
    MZAT12= y1[7]
    ZAT12 = y1[8]
    MCOR15A= y1[9]
    COR15A= y1[10]
    
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
    vs1= 30  
    vs2= 8  
    vs3= 1.4
    Ka1= 0.35
    Ka2= 0.3
    Ka3= 0.6
    Ka4= 0.5
    KI1= 0.21
    KI2= 0.7  
    KI3= 0.3 
    KI4= 0.28
    KI5= 1.8
    KI6= 4
    vd1= 0.1
    vd2= 1.1
    vd3= (t-Tc>0)*0.55+(Tc-t>0)*5.5
    vd4= 1.2
    vd5= 1.8
    vd6= 0.1  
    vd7= 1.6
    vd8= 1.25
    vd9= 0.3
    Kd1= 0.2
    Kd2= 0.4
    Kd3= 0.4  
    Kd4= 0.1
    Kd5= 0.25
    Kd6= 1.5
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

def f2(t,y2):
    Ca    = y2[0]
    MYB15 = y2[1]
    ICE1  = y2[2]
    ICE1P = y2[3]
    MCBF3 = y2[4]
    CBF3  = y2[5]
    CBF3P = y2[6]
    MZAT12= y2[7]
    ZAT12 = y2[8]
    MCOR15A= y2[9]
    COR15A= y2[10]
    
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
    vs1= 25  
    vs2= 8  
    vs3= 1
    Ka1= 0.35
    Ka2= 0.3
    Ka3= 0.6
    Ka4= 0.5
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
    vd7= 0.4
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


def Gauss2s4_1(tspan,y0,h):
    t=np.arange(tspan[0],tspan[1]+h,h,dtype='float32')
    n=len(t)-1
    m=len(y0)  
    y1=np.zeros((m,n+1)) 
    y1[:,0]=y0 
    tol=1e-4
    for i in range(0,n,1):
        y1p=y1[:,i]+c1*h*f1(t[i],y1[:,i])
        y2p=y1[:,i]+c2*h*f1(t[i],y1[:,i])
        y1c=y1[:,i]+h*(a11*f1(t[i]+c1*h,y1p)+a12*f1(t[i]+c2*h,y2p))  
        y2c=y1[:,i]+h*(a21*f1(t[i]+c1*h,y1p)+a22*f1(t[i]+c2*h,y2p))  
        while np.linalg.norm(y1p-y1c,ord=2)+np.linalg.norm(y2p-y2c,ord=2)>tol:
              y1p=y1c; y2p=y2c
              y1c=y1[:,i]+h*(a11*f1(t[i]+c1*h,y1p)+a12*f1(t[i]+c2*h,y2p))  
              y2c=y1[:,i]+h*(a21*f1(t[i]+c1*h,y1p)+a22*f1(t[i]+c2*h,y2p))
        y1[:,i+1]=y1[:,i]+h*(b1*f1(t[i]+c1*h,y1c)+b2*f1(t[i]+c2*h,y2c))  
    return t,y1

def Gauss2s4_2(tspan,y0,h):
    t=np.arange(tspan[0],tspan[1]+h,h,dtype='float32')
    n=len(t)-1
    m=len(y0)  
    y2=np.zeros((m,n+1)) 
    y2[:,0]=y0 
    tol=1e-4
    for i in range(0,n,1):
        y1p=y2[:,i]+c1*h*f2(t[i],y2[:,i])
        y2p=y2[:,i]+c2*h*f2(t[i],y2[:,i])
        y1c=y2[:,i]+h*(a11*f2(t[i]+c1*h,y1p)+a12*f2(t[i]+c2*h,y2p))  
        y2c=y2[:,i]+h*(a21*f2(t[i]+c1*h,y1p)+a22*f2(t[i]+c2*h,y2p))  
        while np.linalg.norm(y1p-y1c,ord=2)+np.linalg.norm(y2p-y2c,ord=2)>tol:
              y1p=y1c; y2p=y2c
              y1c=y2[:,i]+h*(a11*f2(t[i]+c1*h,y1p)+a12*f2(t[i]+c2*h,y2p))  
              y2c=y2[:,i]+h*(a21*f2(t[i]+c1*h,y1p)+a22*f2(t[i]+c2*h,y2p))
        y2[:,i+1]=y2[:,i]+h*(b1*f2(t[i]+c1*h,y1c)+b2*f2(t[i]+c2*h,y2c))  
    return t,y2

if __name__ == '__main__':
    y0 = np.array([0.1,1.88,0.2259,0.44,0,0.0750,0.0059,0.01,0.01,0.08,0.1])
    h=0.05
    t10=np.arange(0,2+h,h)
    y00=np.ones((len(t10),1))*y0
    y00=y00.T
    tspan = np.array([2,26])
    
    
    t,y = Gauss2s4(tspan, y0 ,h)
    t=np.append(t10,t)
    y=np.append(y00,y,axis=1)
    
    t1,y1 = Gauss2s4_1(tspan, y0 ,h)
    t1=np.append(t10,t1)
    y1=np.append(y00,y1,axis=1)
    
    t2,y2 = Gauss2s4_2(tspan, y0 ,h)
    t2=np.append(t10,t2)
    y2=np.append(y00,y2,axis=1)
#print(y[:,-5])

#plotting time courses
    font = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 16,}
    font1 = {'family' : 'Times New Roman',
             'weight' : 'normal',
             'size'   : 14,}
    

    plt.subplot(231)
    plt.plot(t,y[4,:],'k',label='Simulated')
    plt.xlim([0,27])
    plt.ylim([-0.5,4])
    t_CBF3 = [2,3,5,8,11,26]
    y_CBF3 = [0,2.041841004,2.661087866,1.087866109,0.20083682,0.066945607]
    y_CBF3_err = [0.02863,0.15865,0.3010786,0.1268107,0.080083682,0.036945607]
    plt.scatter(t_CBF3,y_CBF3,marker='o',color='tomato',label='Exp.')
    plt.plot(t_CBF3,y_CBF3, linestyle = '-.',color='chocolate')
    plt.errorbar(t_CBF3,y_CBF3,y_CBF3_err,lw = 0,ecolor='tomato',elinewidth=3,ms=7,capsize=3)
    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 10})
    plt.xlabel('Time(h)',font)
    plt.ylabel('CBF3 mRNA in A.thaliana',font1)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    x1 = np.linspace(0,2,2)
    x2 = np.linspace(2,26,2)
    yy1 = 3.8*np.ones(len(x1))
    yy2 = 4*np.ones(len(x1))
    plt.fill_between(x1,yy1,yy2,facecolor='red')
    plt.fill_between(x2,yy1,yy2,facecolor='aliceblue')
          

    plt.subplot(232)
    plt.plot(t1,y1[4,:],'k',label='Simulated')
    plt.xlim([0,27])
    plt.ylim([-0.5,4])
    t_MSlCBF1 = [2,3,4,6,10,18,26]
    y_MSlCBF1 = [0,2.1235,1.12682937,0.380487867,0.129268329,0.129268329,0.09]
    y_MSlCBF1_err = [0.1363,0.31865,0.3110786,0.20563,0.1068107,0.090083682,0.036945607]
    plt.scatter(t_MSlCBF1,y_MSlCBF1,marker='o',color='tomato',label='Exp.')
    plt.plot(t_MSlCBF1,y_MSlCBF1, linestyle = '-.',color='chocolate')
    plt.errorbar(t_MSlCBF1,y_MSlCBF1,y_MSlCBF1_err,lw = 0,ecolor='tomato',elinewidth=3,ms=7,capsize=3)
    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 10})
    plt.xlabel('Time(h)',font)
    plt.ylabel('CBF1 mRNA in S.Lycopersicum',font1)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    x1 = np.linspace(0,2,2)
    x2 = np.linspace(2,26,2)
    yy1 = 3.8*np.ones(len(x1))
    yy2 = 4*np.ones(len(x1))
    plt.fill_between(x1,yy1,yy2,facecolor='red')
    plt.fill_between(x2,yy1,yy2,facecolor='aliceblue')
    
    plt.subplot(233)
    plt.plot(t2,y2[4,:],'k',label='Simulated')
    plt.xlim([0,27])
    plt.ylim([-0.5,4])
    t_MBcCBF3 = [2,2.5,3,4,6,14,26]
    y_MBcCBF3 = [0,0.355096696,2.399999856,1.533417531,0.80541972,0.313606504,0.13]
    y_MBcCBF3_err = [0.0963,0.21865,0.410786,0.30563,0.16068107,0.10083682,0.16945607]
    plt.scatter(t_MBcCBF3,y_MBcCBF3,marker='o',color='tomato',label='Exp.')
    plt.plot(t_MBcCBF3,y_MBcCBF3, linestyle = '-.',color='chocolate')
    plt.errorbar(t_MBcCBF3,y_MBcCBF3,y_MBcCBF3_err,lw = 0,ecolor='tomato',elinewidth=3,ms=7,capsize=3)
    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 10})
    plt.xlabel('Time(h)',font)
    plt.ylabel('CBF1 mRNA in B.rapa',font1)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    x1 = np.linspace(0,2,2)
    x2 = np.linspace(2,26,2)
    yy1 = 3.8*np.ones(len(x1))
    yy2 = 4*np.ones(len(x1))
    plt.fill_between(x1,yy1,yy2,facecolor='red')
    plt.fill_between(x2,yy1,yy2,facecolor='aliceblue')
    
    plt.subplot(234)
    plt.plot(t,y[9,:],'k',label='Simulated')
    plt.xlim([0,27])
    plt.ylim([-0.5,4])
    t_COR = [2.25,2.5,3,4,5,14,26]
    y_COR = [0.08,0.2,0.379136063,0.789598263,1.108685406,2.389680438,2.74]
    y_COR_err = [0.0763,0.15865,0.110786,0.20563,0.16068107,0.10083682,0.2354]
    plt.scatter(t_COR,y_COR,marker='o',color='blue',label='Exp.')
    plt.plot(t_COR,y_COR, linestyle = '-.',color='steelblue')
    plt.errorbar(t_COR,y_COR,y_COR_err,lw = 0,ecolor='blue',elinewidth=3,ms=7,capsize=3)
    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 10})
    plt.xlabel('Time(h)',font)
    plt.ylabel('COR15A mRNA in A.thaliana',font1)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    x1 = np.linspace(0,2,2)
    x2 = np.linspace(2,26,2)
    yy1 = 3.8*np.ones(len(x1))
    yy2 = 4*np.ones(len(x1))
    plt.fill_between(x1,yy1,yy2,facecolor='red')
    plt.fill_between(x2,yy1,yy2,facecolor='aliceblue')

    plt.subplot(235)
    plt.plot(t1,y1[9,:],'k',label='Simulated')
    plt.xlim([0,27])
    plt.ylim([-0.5,4])
    t_MSlCOR15A = [2,3,5,8,11,14,26]
    y_MSlCOR15A = [0.01,0.0403125,0.353541667,0.61625,1.040625,1.95,1.363958333]
    y_MSlCOR15A_err = [0.0163,0.05865,0.210786,0.20563,0.36068107,0.19083682,0.2654]
    plt.scatter(t_MSlCOR15A,y_MSlCOR15A,marker='o',color='blue',label='Exp.')
    plt.plot(t_MSlCOR15A,y_MSlCOR15A, linestyle = '-.',color='steelblue')
    plt.errorbar(t_MSlCOR15A,y_MSlCOR15A,y_MSlCOR15A_err,lw = 0,ecolor='blue',elinewidth=3,ms=7,capsize=3)
    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 10})
    plt.xlabel('Time(h)',font)
    plt.ylabel('COR413 mRNA in S.Lycopersicum',font1)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    x1 = np.linspace(0,2,2)
    x2 = np.linspace(2,26,2)
    yy1 = 3.8*np.ones(len(x1))
    yy2 = 4*np.ones(len(x1))
    plt.fill_between(x1,yy1,yy2,facecolor='red')
    plt.fill_between(x2,yy1,yy2,facecolor='aliceblue')
    
    
    plt.subplot(236)
    plt.plot(t2,y2[9,:],'k',label='Simulated')
    plt.xlim([0,27])
    plt.ylim([-0.5,4])
    t_MBcCOR15A = [2.5,3,4,6,14,26]
    y_MBcCOR15A = [0.08,0.089066668,0.088978352,0.96,1.759421438,2.58]
    y_MBcCOR15A_err = [0.0563,0.06865,0.100786,0.22563,0.36068107,0.19083682]
    plt.scatter(t_MBcCOR15A,y_MBcCOR15A,marker='o',color='blue',label='Exp.')
    plt.plot(t_MBcCOR15A,y_MBcCOR15A, linestyle = '-.',color='steelblue')
    plt.errorbar(t_MBcCOR15A,y_MBcCOR15A,y_MBcCOR15A_err,lw = 0,ecolor='blue',elinewidth=3,ms=7,capsize=3)
    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 10})
    plt.xlabel('Time(h)',font)
    plt.ylabel('COR15A mRNA in B.rapa',font1)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    x1 = np.linspace(0,2,2)
    x2 = np.linspace(2,26,2)
    yy1 = 3.8*np.ones(len(x1))
    yy2 = 4*np.ones(len(x1))
    plt.fill_between(x1,yy1,yy2,facecolor='red')
    plt.fill_between(x2,yy1,yy2,facecolor='aliceblue')
    
    plt.show()