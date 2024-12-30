# -*- coding: utf-8 -*-
"""
Created on Mon May 17 20:23:44 2021

@author: Huangtin
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from scipy.signal import find_peaks
from scipy import signal
import pandas as pd



def f1(t,y,Tc):

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
#    Tc = 2
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
    vs1= 20    #34
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
    vd4= 0.9  #1.2
    vd5= 1.8
    vd6= 0.1  
    vd7= 1.3
    vd8= (t-Tc>0)*1.5+(Tc-t>0)*1.25
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
                 vs1*ICE1**n/(KI1**n+ICE1**n)*KI2**m/(KI2**m+MYB15**m)*KI3**r/(KI3**r+ZAT12**r)-vd3*MCBF3/(Kd3+MCBF3),\
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


def Gauss2s4_1(tspan,y0,h,Tc):
    t=np.arange(tspan[0],tspan[1]+h,h,dtype='float32')
    n=len(t)-1
    m=len(y0)  
    y=np.zeros((m,n+1)) 
    y[:,0]=y0 
    tol=1e-4
    for i in range(0,n,1):
        y1p=y[:,i]+c1*h*f1(t[i],y[:,i],Tc)
        y2p=y[:,i]+c2*h*f1(t[i],y[:,i],Tc)
        y1c=y[:,i]+h*(a11*f1(t[i]+c1*h,y1p,Tc)+a12*f1(t[i]+c2*h,y2p,Tc))  
        y2c=y[:,i]+h*(a21*f1(t[i]+c1*h,y1p,Tc)+a22*f1(t[i]+c2*h,y2p,Tc))  
        while np.linalg.norm(y1p-y1c,ord=2)+np.linalg.norm(y2p-y2c,ord=2)>tol:
              y1p=y1c; y2p=y2c
              y1c=y[:,i]+h*(a11*f1(t[i]+c1*h,y1p,Tc)+a12*f1(t[i]+c2*h,y2p,Tc))  
              y2c=y[:,i]+h*(a21*f1(t[i]+c1*h,y1p,Tc)+a22*f1(t[i]+c2*h,y2p,Tc))
        y[:,i+1]=y[:,i]+h*(b1*f1(t[i]+c1*h,y1c,Tc)+b2*f1(t[i]+c2*h,y2c,Tc))  
    return t,y




if __name__ == '__main__':
    
 #coefficient of funCa  
    
    
    h=0.005
    
    y0 = np.array([0.1,1.88,0.2259,0.44,0.08107378861066673,0.0750,0.0059,0.04705186316759576,0.01,0.08,0.1])

    Tc1 = 2
    Tc2 = 4
    Tc3 = 12
    
    
    
    t10=np.arange(0,2+h,h)
    tspan1 = np.array([2,600])
    y00=np.ones((len(t10),1))*y0
    y00=y00.T
    t1,y1 = Gauss2s4_1(tspan1, y0 ,h,Tc1)
    t1=np.append(t10,t1)
    y1=np.append(y00,y1,axis=1)
    
    
#    t20=np.arange(0,4+h,h)
#    tspan2 = np.array([4,96])
#    y00=np.ones((len(t20),1))*y0
#    y00=y00.T
#    t2,y2 = Gauss2s4_2(tspan2, y0 ,h,Tc1)
#    t2=np.append(t20,t2)
#    y2=np.append(y00,y2,axis=1)
#    
#    
#    yval1 = y2[4,9600:120000]
#    ampUp1 = []
#    ampLow1 = []
#    for i in range(1, len(yval1)-1):
#        if(yval1[i-1] < yval1[i] and yval1[i+1] < yval1[i]):
#            ampUp1.append(yval1[i])
#        elif(yval1[i-1] > yval1[i] and yval1[i+1] > yval1[i]):
#            ampLow1.append(yval1[i])
#        else:
#            pass
#
#        
#        
#    yval2 = y2[9,9600:120000]
#    ampUp2 = []
#    ampLow2 = []
#    for i in range(1, len(yval2)-1):
#        if(yval2[i-1] < yval2[i] and yval2[i+1] < yval2[i]):
#            ampUp2.append(yval2[i])
#        elif(yval2[i-1] > yval2[i] and yval2[i+1] > yval2[i]):
#            ampLow2.append(yval2[i])
#        else:
#            pass
#
#  
#    T1=[]
#    phase1 = []
#    peak_id1,peak_property1 = find_peaks(y2[4,4800:60000], height=0, distance=20)
#    T1.append(np.mean(np.diff(peak_id1)/100))
#    phase1.append(t1[peak_id1])
#    
#    
#    T2=[]
#    amplitude2=[]
#    phase2 = []
#    peak_id2,peak_property2 = find_peaks(y2[9,4800:60000], height=0, distance=20)
#    T2.append(np.mean(np.diff(peak_id1)/100))
#    phase2.append(t2[peak_id2])
#    
#    t30=np.arange(0,12+h,h)
#    tspan3 = np.array([12,100])
#    y00=np.ones((len(t30),1))*y0
#    y00=y00.T
#    t3,y3 = Gauss2s4_1(tspan3, y0 ,h,Tc3)
#    t3=np.append(t30,t3)
#    y3=np.append(y00,y3,axis=1)
    
        
    font = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 16,}
    font1 = {'family' : 'Times New Roman',
             'weight' : 'normal',
             'size'   : 14,}



    plt.subplot(221)
    
    plt.plot(t1,y1[4,:],'k',label='CBF3 mRNA')
    plt.plot(t1,y1[9,:],'r',label='COR15A mRNA')
    plt.xlim([0,96])
    plt.ylim([0,2.5])
    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 10})
    plt.xlabel('Time(h)',font)
    plt.ylabel('CBF3 and COR15A mRNA',font1)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    p1 = np.linspace(0,2,2)
    p2 = np.linspace(2,96,2)
    yyy1 = 2.4*np.ones(len(p1))
    yyy2 = 2.5*np.ones(len(p1))
    plt.fill_between(p1,yyy1,yyy2,facecolor='red')
    plt.fill_between(p2,yyy1,yyy2,facecolor='aliceblue')
    x_major_locator=MultipleLocator(12)
    y_major_locator=MultipleLocator(0.5)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
#    
    
#    plt.subplot(122)
#    
#    plt.plot(t2,y2[4,:],'k',label='CBF3 mRNA')
#    plt.plot(t2,y2[9,:],'r',label='COR15A mRNA')
#    plt.xlim([0,96])
#    plt.ylim([0,2.5])
#    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 10})
#    plt.xlabel('Time(h)',font)
#    plt.ylabel('CBF3 and COR15A mRNA',font1)
#    plt.xticks(fontproperties = 'Times New Roman', size = 14)
#    plt.yticks(fontproperties = 'Times New Roman', size = 14)
#    p1 = np.linspace(0,4,2)
#    p2 = np.linspace(4,96,2)
#    yyy1 = 2.4*np.ones(len(p1))
#    yyy2 = 2.5*np.ones(len(p1))
#    plt.fill_between(p1,yyy1,yyy2,facecolor='red')
#    plt.fill_between(p2,yyy1,yyy2,facecolor='aliceblue')
#    x_major_locator=MultipleLocator(12)
#    y_major_locator=MultipleLocator(0.5)
#    ax=plt.gca()
#    ax.xaxis.set_major_locator(x_major_locator)
#    ax.yaxis.set_major_locator(y_major_locator)
    
    plt.subplot(222)
    file1 = pd.read_csv('per_phase_sen.csv', header = None, sep = ',') 

    df1 = pd.DataFrame(file1)

    kCa1 = np.array(df1[0])
    per = np.array(df1[1])
    phase1= np.array(df1[2])
    phase2 = np.array(df1[3])
    cor = df1[0]




    plt.scatter(kCa1,per,s=12,marker='o',color='black',label='period')
    plt.scatter(kCa1,phase1,s=18,marker='+',color='aqua',label='peaktime of CBF3')
    plt.scatter(kCa1,phase2,s=18,marker='*',color='lime',label='peaktime of COR15A')
#plt.vlines(0.14, 20, 125,color='red')
    plt.scatter(0.14,23.7475,s=28,marker='o',color='red')
    plt.scatter(0.14,42.4901,s=34,marker='+',color='red')
    plt.scatter(0.14,48.2752,s=34,marker='*',color='red')
    plt.xlim(0.125,0.171)
    plt.ylim(20,125)
    plt.xlabel('$KC_{a}$',font)
    plt.ylabel('period, the first peaktime of CBF3 and COR15A mRNA',font1)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 12})
    x_major_locator=MultipleLocator(0.01)
    y_major_locator=MultipleLocator(20)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    
    
    file1 = pd.read_csv('amp_sen_CBF.csv', header = None, sep = ',') 
    file2 = pd.read_csv('amp_sen_COR.csv', header = None, sep = ',') 
    df1 = pd.DataFrame(file1)
    df2 = pd.DataFrame(file2)
    kCa1 = np.array(df1[0])
    kCa2 = np.array(df2[0])
    CBF_equi = np.array(df1[1])
    CBF_up = np.array(df1[2])
    CBF_low = np.array(df1[3])
    COR_equi = np.array(df2[1])
    COR_up = np.array(df2[2])
    COR_low = np.array(df2[3])


    plt.subplot(223)
    plt.plot(kCa1[0:4],CBF_equi[0:4], linestyle = '-',color='black')
    plt.plot(kCa1[3:],CBF_equi[3:], linestyle = '--',color='black')
    plt.scatter(kCa1[3:],CBF_up[3:],s=12,marker='o',color='blue')
    plt.scatter(kCa1[3:],CBF_low[3:],s=12,marker='o',color='red')
    plt.xlim(0.12,0.17)
    plt.ylim(-0.1,2)
    plt.xlabel('$KC_{a}$',font)
    plt.ylabel('CBF3 transcript level',font1)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)

    x_major_locator=MultipleLocator(0.01)
    y_major_locator=MultipleLocator(0.5)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)


    plt.subplot(224)
    plt.plot(kCa2[0:4],COR_equi[0:4], linestyle = '-',color='black')
    plt.plot(kCa2[3:],COR_equi[3:], linestyle = '--',color='black')
    plt.scatter(kCa2[3:],COR_up[3:],s=12,marker='o',color='blue')
    plt.scatter(kCa2[3:],COR_low[3:],s=12,marker='o',color='red')
    plt.xlim(0.12,0.3)
    plt.ylim(-0.1,2)
    plt.xlabel('$KC_{a}$',font)
    plt.ylabel('COR15A transcript level',font1)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)

    x_major_locator=MultipleLocator(0.04)
    y_major_locator=MultipleLocator(0.5)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)

    
