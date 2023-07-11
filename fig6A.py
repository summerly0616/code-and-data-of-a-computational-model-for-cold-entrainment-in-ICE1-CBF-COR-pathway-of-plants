# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 09:02:37 2021

@author: Huangting
"""

import numpy as np
import matplotlib.pyplot as plt

def f1(t,y):
    KCa = 0.265
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
    Tp = 1
    Td =0.1
    Tc1 = 2
    Tc2 = 341
    
    Iin0 = 0.28
    Iin1 = 20
    Iex0 = 2.8
    Iin = (t-Tc1>0)*(Tc1+Td-t>0)*Iin1+(t-Tc2>0)*(Tc2+Td-t>0)*Iin1  
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
    vd3= ((t-Tc1>0)*(4-t>0)+(t-4>0)*(340-t>0)+(t-Tc2>0))*0.55+((t>0)*(Tc1-t>0)+(t>340)*(Tc2-t>0))*5.5
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
                 ks1-k2*CaM/(Ka1+CaM)*ICE1/(K2+ICE1)*Ka2/(Ka2+CaM)+v2*ICE1P/(K2P+ICE1P)-vd1*ICE1/(Kd1+ICE1)*(Ka4/(Ka4+CaM)*((t-Tc1>0)*(4-t>0)+(t-4>0)*(340-t>0)+(t-Tc2>0)*(Tc2+Tp-t>0))+1*((t>0)*(Tc1-t>0)+(t>340)*(Tc2-t>0))),\
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
#    print(t)
    KCa = 0.265
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
    Td =0.1
    Tc1 = 2
    Tc2 = 343
    
    Iin0 = 0.28
    Iin1 = 20
    Iex0 = 2.8
    Iin = (t-Tc1>0)*(Tc1+Td-t>0)*Iin1+(t-Tc2>0)*(Tc2+Td-t>0)*Iin1  
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
    vd3= ((t-Tc1>0)*(4-t>0)+(t-4>0)*(340-t>0)+(t-Tc2>0))*0.55+((t>0)*(Tc1-t>0)+(t>340)*(Tc2-t>0))*5.5
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
                 ks1-k2*CaM/(Ka1+CaM)*ICE1/(K2+ICE1)*Ka2/(Ka2+CaM)+v2*ICE1P/(K2P+ICE1P)-vd1*ICE1/(Kd1+ICE1)*(Ka4/(Ka4+CaM)*((t-Tc1>0)*(4-t>0)+(t-4>0)*(340-t>0)+(t-Tc2>0)*(Tc2+Tp-t>0))+1*((t>0)*(Tc1-t>0)+(t>340)*(Tc2-t>0))),\
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
#    print(t)
    KCa = 0.265
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
    Td =0.1
    Tc1 = 2
    Tc2 = 346
    
    Iin0 = 0.28
    Iin1 = 20
    Iex0 = 2.8
    Iin = (t-Tc1>0)*(Tc1+Td-t>0)*Iin1+(t-Tc2>0)*(Tc2+Td-t>0)*Iin1  
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
    vd3= ((t-Tc1>0)*(4-t>0)+(t-4>0)*(340-t>0)+(t-Tc2>0))*0.55+((t>0)*(Tc1-t>0)+(t>340)*(Tc2-t>0))*5.5
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
                 ks1-k2*CaM/(Ka1+CaM)*ICE1/(K2+ICE1)*Ka2/(Ka2+CaM)+v2*ICE1P/(K2P+ICE1P)-vd1*ICE1/(Kd1+ICE1)*(Ka4/(Ka4+CaM)*((t-Tc1>0)*(4-t>0)+(t-4>0)*(340-t>0)+(t-Tc2>0)*(Tc2+Tp-t>0))+1*((t>0)*(Tc1-t>0)+(t>340)*(Tc2-t>0))),\
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
#    print(t)
    KCa = 0.27
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
    Tp = 8
    Td =0.1
    Tc1 = 2
    Tc2 = 348
    
    Iin0 = 0.28
    Iin1 = 20
    Iex0 = 2.8
    Iin = (t-Tc1>0)*(Tc1+Td-t>0)*Iin1+(t-Tc2>0)*(Tc2+Td-t>0)*Iin1  
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
    vd3= ((t-Tc1>0)*(4-t>0)+(t-4>0)*(340-t>0)+(t-Tc2>0))*0.55+((t>0)*(Tc1-t>0)+(t>340)*(Tc2-t>0))*5.5
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
                 ks1-k2*CaM/(Ka1+CaM)*ICE1/(K2+ICE1)*Ka2/(Ka2+CaM)+v2*ICE1P/(K2P+ICE1P)-vd1*ICE1/(Kd1+ICE1)*(Ka4/(Ka4+CaM)*((t-Tc1>0)*(4-t>0)+(t-4>0)*(340-t>0)+(t-Tc2>0)*(Tc2+Tp-t>0))+1*((t>0)*(Tc1-t>0)+(t>340)*(Tc2-t>0))),\
                 k2*CaM/(Ka1+CaM)*ICE1/(K2+ICE1)*Ka2/(Ka2+CaM)-v2*ICE1P/(K2P+ICE1P)-vd2*ICE1P/(Kd2+ICE1P)*CaM/(Ka3+CaM),\
                 vs1*CaM*ICE1**n/(KI1**n+ICE1**n)*KI2**m/(KI2**m+MYB15**m)*KI3**r/(KI3**r+ZAT12**r)-vd3*MCBF3/(Kd3+MCBF3),\
                 ks2*MCBF3-k3*CaM*CBF3/(K3+CBF3)+v3*CBF3P/(K3P+CBF3P)-vd4*CBF3/(Kd4+CBF3),\
                 k3*CaM*CBF3/(K3+CBF3)-v3*CBF3P/(K3P+CBF3P)-vd5*CBF3P/(Kd5+CBF3P),\
                 vs2*CBF3**s/(KI4**s+CBF3**s)-vd6*MZAT12/(Kd6+MZAT12),\
                 ks3*MZAT12-vd7*ZAT12/(Kd7+ZAT12),\
                 vs3*(1-1/(1+(CBF3/KI5)**u+(ZAT12/KI6)**v))-vd8*MCOR15A/(Kd8+MCOR15A),\
                 ks4*MCOR15A-vd9*COR15A/(Kd9+COR15A)])
    return z

def f5(t,y):
#    print(t)
    KCa = 0.31
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
    Tp = 24
    Td =0.1
    Tc1 = 2
    Tc2 = 364
    
    Iin0 = 0.28
    Iin1 = 20
    Iex0 = 2.8
    Iin = (t-Tc1>0)*(Tc1+Td-t>0)*Iin1+(t-Tc2>0)*(Tc2+Td-t>0)*Iin1  
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
    vd3= ((t-Tc1>0)*(4-t>0)+(t-4>0)*(340-t>0)+(t-Tc2>0))*0.55+((t>0)*(Tc1-t>0)+(t>340)*(Tc2-t>0))*5.5
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
                 ks1-k2*CaM/(Ka1+CaM)*ICE1/(K2+ICE1)*Ka2/(Ka2+CaM)+v2*ICE1P/(K2P+ICE1P)-vd1*ICE1/(Kd1+ICE1)*(Ka4/(Ka4+CaM)*((t-Tc1>0)*(4-t>0)+(t-4>0)*(340-t>0)+(t-Tc2>0)*(Tc2+Tp-t>0))+1*((t>0)*(Tc1-t>0)+(t>340)*(Tc2-t>0))),\
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

def Gauss2s4_5(tspan,y0,h):
    t=np.arange(tspan[0],tspan[1]+h,h,dtype='float32')
    n=len(t)-1
    m=len(y0)  
    y=np.zeros((m,n+1)) 
    y[:,0]=y0 
    tol=1e-4
    for i in range(0,n,1):
        y1p=y[:,i]+c1*h*f5(t[i],y[:,i])
        y2p=y[:,i]+c2*h*f5(t[i],y[:,i])
        y1c=y[:,i]+h*(a11*f5(t[i]+c1*h,y1p)+a12*f5(t[i]+c2*h,y2p))  
        y2c=y[:,i]+h*(a21*f5(t[i]+c1*h,y1p)+a22*f5(t[i]+c2*h,y2p))  
        while np.linalg.norm(y1p-y1c,ord=2)+np.linalg.norm(y2p-y2c,ord=2)>tol:
              y1p=y1c; y2p=y2c
              y1c=y[:,i]+h*(a11*f5(t[i]+c1*h,y1p)+a12*f5(t[i]+c2*h,y2p))  
              y2c=y[:,i]+h*(a21*f5(t[i]+c1*h,y1p)+a22*f5(t[i]+c2*h,y2p))
        y[:,i+1]=y[:,i]+h*(b1*f5(t[i]+c1*h,y1c)+b2*f5(t[i]+c2*h,y2c))  
    return t,y



y0 = np.array([0.1,1.92,0.2259,0.2981,0.01,0.0750,0.0059,0.05,0.05,0,0.01])
y02 = np.array([0.1,1.5,0.2,0.2981,0.01,0.0750,0.0059,0.05,0.05,0,0.01])
h=0.05
    
t10=np.arange(0,2+h,h)
y10=np.ones((len(t10),1))*y0
y10=y10.T
tspan1 = np.array([2,388])
t1,y1 = Gauss2s4_1(tspan1, y0 ,h)
t1=np.append(t10,t1)
y1=np.append(y10,y1,axis=1)
     
t20=np.arange(0,2+h,h)
y20=np.ones((len(t20),1))*y0
y20=y20.T
tspan2 = np.array([2,388]) 
t2,y2 = Gauss2s4_2(tspan2, y0 ,h)
t2=np.append(t20,t2)
y2=np.append(y20,y2,axis=1)    
#    
t30=np.arange(0,2+h,h)
y30=np.ones((len(t30),1))*y0
y30=y30.T
tspan3 = np.array([2,388])
t3,y3 = Gauss2s4_3(tspan3, y0 ,h)
t3=np.append(t30,t3)
y3=np.append(y30,y3,axis=1)


t40=np.arange(0,2+h,h)
y40=np.ones((len(t40),1))*y0
y40=y40.T
tspan4 = np.array([2,388])
t4,y4 = Gauss2s4_4(tspan4, y0 ,h)
t4=np.append(t40,t4)
y4=np.append(y40,y4,axis=1)


t50=np.arange(0,2+h,h)
y50=np.ones((len(t50),1))*y0
y50=y50.T
tspan5 = np.array([2,388])
t5,y5 = Gauss2s4_5(tspan5, y0 ,h)
t5=np.append(t50,t5)
y5=np.append(y50,y5,axis=1)

font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 16,}
font1 = {'family' : 'Times New Roman',
         'weight' : 'normal',
         'size'   : 14,}


plt.subplot(521)
plt.plot(t1,y1[4,:],color='black',linestyle='-',label='CBF3 mRNA',linewidth=1.5)

plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.yticks(fontproperties = 'Times New Roman', size = 14)
plt.ylim([0,2])
plt.xlim([0,15])
tt1 = np.linspace(0,2,2)
x1 = 1.8*np.ones(len(tt1))
x2 = 2*np.ones(len(tt1))
tt2 = np.linspace(2,340,2)
x3 = 1.8*np.ones(len(tt2))
x4 = 2*np.ones(len(tt2))
tt3 = np.linspace(340,341,2)
x5 = 1.8*np.ones(len(tt3))
x6 = 2*np.ones(len(tt3))
tt4 = np.linspace(341,388,2)
x7 = 1.8*np.ones(len(tt4))
x8 = 2*np.ones(len(tt4))
plt.fill_between(tt1,x1,x2,facecolor='red')
plt.fill_between(tt2,x3,x4,facecolor='aliceblue')
plt.fill_between(tt3,x5,x6,facecolor='red')
plt.fill_between(tt4,x7,x8,facecolor='aliceblue')
plt.xticks([])

plt.subplot(522)
plt.plot(t1,y1[4,:],color='black',linestyle='-',label='CBF3 mRNA',linewidth=1.5)

plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.yticks(fontproperties = 'Times New Roman', size = 14)
plt.ylim([0,2])
plt.xlim([340,388])
tt1 = np.linspace(0,2,2)
x1 = 1.8*np.ones(len(tt1))
x2 = 2*np.ones(len(tt1))
tt2 = np.linspace(2,340,2)
x3 = 1.8*np.ones(len(tt2))
x4 = 2*np.ones(len(tt2))
tt3 = np.linspace(340,341,2)
x5 = 1.8*np.ones(len(tt3))
x6 = 2*np.ones(len(tt3))
tt4 = np.linspace(341,388,2)
x7 = 1.8*np.ones(len(tt4))
x8 = 2*np.ones(len(tt4))
plt.fill_between(tt1,x1,x2,facecolor='red')
plt.fill_between(tt2,x3,x4,facecolor='aliceblue')
plt.fill_between(tt3,x5,x6,facecolor='red')
plt.fill_between(tt4,x7,x8,facecolor='aliceblue')
plt.xticks([])
plt.yticks([])



plt.subplot(523)
plt.plot(t2,y2[4,:],color='black',linestyle='-',label='CBF3 mRNA',linewidth=1.5)

plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.yticks(fontproperties = 'Times New Roman', size = 14)
plt.ylim([0,2])
plt.xlim([0,15])
tt1 = np.linspace(0,2,2)
x1 = 1.8*np.ones(len(tt1))
x2 = 2*np.ones(len(tt1))
tt2 = np.linspace(2,340,2)
x3 = 1.8*np.ones(len(tt2))
x4 = 2*np.ones(len(tt2))
tt3 = np.linspace(340,343,2)
x5 = 1.8*np.ones(len(tt3))
x6 = 2*np.ones(len(tt3))
tt4 = np.linspace(343,388,2)
x7 = 1.8*np.ones(len(tt4))
x8 = 2*np.ones(len(tt4))
plt.fill_between(tt1,x1,x2,facecolor='red')
plt.fill_between(tt2,x3,x4,facecolor='aliceblue')
plt.fill_between(tt3,x5,x6,facecolor='red')
plt.fill_between(tt4,x7,x8,facecolor='aliceblue')
plt.xticks([])

plt.subplot(524)
plt.plot(t2,y2[4,:],color='black',linestyle='-',label='CBF3 mRNA',linewidth=1.5)

plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.yticks(fontproperties = 'Times New Roman', size = 14)
plt.ylim([0,2])
plt.xlim([340,388])
tt1 = np.linspace(0,2,2)
x1 = 1.8*np.ones(len(tt1))
x2 = 2*np.ones(len(tt1))
tt2 = np.linspace(2,340,2)
x3 = 1.8*np.ones(len(tt2))
x4 = 2*np.ones(len(tt2))
tt3 = np.linspace(340,343,2)
x5 = 1.8*np.ones(len(tt3))
x6 = 2*np.ones(len(tt3))
tt4 = np.linspace(343,388,2)
x7 = 1.8*np.ones(len(tt4))
x8 = 2*np.ones(len(tt4))
plt.fill_between(tt1,x1,x2,facecolor='red')
plt.fill_between(tt2,x3,x4,facecolor='aliceblue')
plt.fill_between(tt3,x5,x6,facecolor='red')
plt.fill_between(tt4,x7,x8,facecolor='aliceblue')
plt.xticks([])
plt.yticks([])


plt.subplot(525)
plt.plot(t3,y3[4,:],color='black',linestyle='-',label='CBF3 mRNA',linewidth=1.5)
plt.ylabel('relative expression of CBF3 mRNA',font)

plt.xlim([0,15])
plt.ylim([0,2])
plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.yticks(fontproperties = 'Times New Roman', size = 14)
tt1 = np.linspace(0,2,2)
x1 = 1.8*np.ones(len(tt1))
x2 = 2*np.ones(len(tt1))
tt2 = np.linspace(2,340,2)
x3 = 1.8*np.ones(len(tt2))
x4 = 2*np.ones(len(tt2))
tt3 = np.linspace(340,346,2)
x5 = 1.8*np.ones(len(tt3))
x6 = 2*np.ones(len(tt3))
tt4 = np.linspace(346,388,2)
x7 = 1.8*np.ones(len(tt4))
x8 = 2*np.ones(len(tt4))
plt.fill_between(tt1,x1,x2,facecolor='red')
plt.fill_between(tt2,x3,x4,facecolor='aliceblue')
plt.fill_between(tt3,x5,x6,facecolor='red')
plt.fill_between(tt4,x7,x8,facecolor='aliceblue')
plt.xticks([])


plt.subplot(526)
plt.plot(t3,y3[4,:],color='black',linestyle='-',label='CBF3 mRNA',linewidth=1.5)

plt.xlim([340,388])
plt.ylim([0,2])
plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.yticks(fontproperties = 'Times New Roman', size = 14)
tt1 = np.linspace(0,2,2)
x1 = 1.8*np.ones(len(tt1))
x2 = 2*np.ones(len(tt1))
tt2 = np.linspace(2,340,2)
x3 = 1.8*np.ones(len(tt2))
x4 = 2*np.ones(len(tt2))
tt3 = np.linspace(340,346,2)
x5 = 1.8*np.ones(len(tt3))
x6 = 2*np.ones(len(tt3))
tt4 = np.linspace(346,388,2)
x7 = 1.8*np.ones(len(tt4))
x8 = 2*np.ones(len(tt4))
plt.fill_between(tt1,x1,x2,facecolor='red')
plt.fill_between(tt2,x3,x4,facecolor='aliceblue')
plt.fill_between(tt3,x5,x6,facecolor='red')
plt.fill_between(tt4,x7,x8,facecolor='aliceblue')
plt.xticks([])
plt.yticks([])



plt.subplot(527)
plt.plot(t4,y4[4,:],color='black',linestyle='-',label='CBF3 mRNA',linewidth=1.5)

plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.yticks(fontproperties = 'Times New Roman', size = 14)
plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.yticks(fontproperties = 'Times New Roman', size = 14)
plt.ylim([0,2])
plt.xlim([0,15])
tt1 = np.linspace(0,2,2)
x1 = 1.8*np.ones(len(tt1))
x2 = 2*np.ones(len(tt1))
tt2 = np.linspace(2,340,2)
x3 = 1.8*np.ones(len(tt2))
x4 = 2*np.ones(len(tt2))
tt3 = np.linspace(340,348,2)
x5 = 1.8*np.ones(len(tt3))
x6 = 2*np.ones(len(tt3))
tt4 = np.linspace(348,388,2)
x7 = 1.8*np.ones(len(tt4))
x8 = 2*np.ones(len(tt4))
plt.fill_between(tt1,x1,x2,facecolor='red')
plt.fill_between(tt2,x3,x4,facecolor='aliceblue')
plt.fill_between(tt3,x5,x6,facecolor='red')
plt.fill_between(tt4,x7,x8,facecolor='aliceblue')
plt.xticks([])


plt.subplot(528)
plt.plot(t4,y4[4,:],color='black',linestyle='-',label='CBF3 mRNA',linewidth=1.5)

plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.yticks(fontproperties = 'Times New Roman', size = 14)
plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.yticks(fontproperties = 'Times New Roman', size = 14)
plt.ylim([0,2])
plt.xlim([340,388])
tt1 = np.linspace(0,2,2)
x1 = 1.8*np.ones(len(tt1))
x2 = 2*np.ones(len(tt1))
tt2 = np.linspace(2,340,2)
x3 = 1.8*np.ones(len(tt2))
x4 = 2*np.ones(len(tt2))
tt3 = np.linspace(340,348,2)
x5 = 1.8*np.ones(len(tt3))
x6 = 2*np.ones(len(tt3))
tt4 = np.linspace(348,388,2)
x7 = 1.8*np.ones(len(tt4))
x8 = 2*np.ones(len(tt4))
plt.fill_between(tt1,x1,x2,facecolor='red')
plt.fill_between(tt2,x3,x4,facecolor='aliceblue')
plt.fill_between(tt3,x5,x6,facecolor='red')
plt.fill_between(tt4,x7,x8,facecolor='aliceblue')
plt.xticks([])
plt.yticks([])


plt.subplot(529)
plt.plot(t4,y4[4,:],color='black',linestyle='-',label='CBF3 mRNA',linewidth=1.5)
plt.ylim([0,2])
plt.xlim([0,15])
plt.xlabel('Time(h)',font)
plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.yticks(fontproperties = 'Times New Roman', size = 14)
tt1 = np.linspace(0,2,2)
x1 = 1.8*np.ones(len(tt1))
x2 = 2*np.ones(len(tt1))
tt2 = np.linspace(2,340,2)
x3 = 1.8*np.ones(len(tt2))
x4 = 2*np.ones(len(tt2))
tt3 = np.linspace(340,364,2)
x5 = 1.8*np.ones(len(tt3))
x6 = 2*np.ones(len(tt3))
tt4 = np.linspace(364,388,2)
x7 = 1.8*np.ones(len(tt4))
x8 = 2*np.ones(len(tt4))
plt.fill_between(tt1,x1,x2,facecolor='red')
plt.fill_between(tt2,x3,x4,facecolor='aliceblue')
plt.fill_between(tt3,x5,x6,facecolor='red')
plt.fill_between(tt4,x7,x8,facecolor='aliceblue')


plt.subplot(5,2,10)
plt.plot(t5,y5[4,:],color='black',linestyle='-',label='CBF3 mRNA',linewidth=1.5)
plt.ylim([0,2])
plt.xlim([340,388])
plt.xlabel('Time(h)',font)
plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.yticks(fontproperties = 'Times New Roman', size = 14)
tt1 = np.linspace(0,2,2)
x1 = 1.8*np.ones(len(tt1))
x2 = 2*np.ones(len(tt1))
tt2 = np.linspace(2,340,2)
x3 = 1.8*np.ones(len(tt2))
x4 = 2*np.ones(len(tt2))
tt3 = np.linspace(340,364,2)
x5 = 1.8*np.ones(len(tt3))
x6 = 2*np.ones(len(tt3))
tt4 = np.linspace(364,388,2)
x7 = 1.8*np.ones(len(tt4))
x8 = 2*np.ones(len(tt4))
plt.fill_between(tt1,x1,x2,facecolor='red')
plt.fill_between(tt2,x3,x4,facecolor='aliceblue')
plt.fill_between(tt3,x5,x6,facecolor='red')
plt.fill_between(tt4,x7,x8,facecolor='aliceblue')
plt.yticks([])

plt.show()