# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 10:14:58 2021

@author: Huangting
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import pandas as pd

plt.rcParams['font.family'] = 'Times New Roman'
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
    
    KCa = 0.18
    Tc = 4
    Tc1 = 55
    Td = 0.05
    Td1 = 0.3
    Iin0 = 0.28
    Iin1 = (t-Tc>0)*(Tc+Td-t>0)*20+(t-Tc1>0)*(Tc1+Td1-t>0)*30
    Iex0 = 2.8
    Iin =(t-Tc>0)*(Tc+Td-t>0)*Iin1+(t-Tc1>0)*(Tc1+Td1-t>0)*Iin1

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
    vs2= 9.5  
    vs3= (t-Tc>0)*(t<Tc1)*1+(Tc-t>0)*1+(t-Tc1>0)*1
    Ka1= 0.35
    Ka2= 0.3
    Ka3= 0.6
    Ka4= 0.5
    KI1= 0.21
    KI2= 0.8  
    KI3= 0.3 
    KI4= 0.28
    KI5= 0.1
    KI6= 0.1
    vd1= 0.1
    vd2= 1.1
    vd3= (t-Tc>0)*(t<Tc1)*0.55+(Tc-t>0)*5.5+(t-Tc1>0)*0.55
    vd4= 1.2
    vd5= 1.8
    vd6= 0.1  
    vd7= 1.3
    vd8= 1.3
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
#    kd = 0.05

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


if __name__ == '__main__':
    y0 = np.array([0.1,1.88,0.2259,0.44,0.1,0.0750,0.0059,0.01,0.01,0.08,0.1])
    y01 = np.array([0.1,1.92,0.2259,0.44,0.05,0.0750,0.0059,0.01,0,0.08,0.1])
    h=0.005
    t10=np.arange(0,4+h,h)
    y00=np.ones((len(t10),1))*y0
    y00=y00.T
    tspan = np.array([4,80])
    tspan1 = np.array([0,80])
    
    
    
    t,y = Gauss2s4(tspan, y0 ,h)
    t=np.append(t10,t)
    y=np.append(y00,y,axis=1)
    
    t1,y1 = Gauss2s4(tspan1, y01 ,h)
#print(y[:,-5])
    KCa = 0.14
    n0 = 4
#plotting time courses
    font = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 16,}
    font1 = {'family' : 'Times New Roman',
             'weight' : 'normal',
             'size'   : 14,}
    
    t_ex = [0,1,2,3,4,5,6,9,29,53,54,55,56,57,61,80]
    file = pd.read_csv('tem.csv', header = None, sep = ',') 
    df = pd.DataFrame(file)
    par = df[0]
    tt = df[1]

    plt.subplot(131)
    plt.plot(par,tt,'k')

    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xlabel('Time(h)',font)
    plt.ylabel('Temperature(℃)',font)
    plt.xlim(0,80)
    plt.ylim(0,20)

    x_major_locator=MultipleLocator(20)
    y_major_locator=MultipleLocator(2)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    
    
    
    plt.subplot(132)
    plt.plot(t,y[4,:],color='black',label='Sim.')
    y_CBF = [0.1,0.220831273,0.182248799,0.510447288,0.861440539,1.309087401,1.592979707,0.820158379,0.539120539,0.366719881,0.853023884,1.909017548,4.072484041,6.72,4.15439943,1.763190904]
    plt.scatter(t_ex,y_CBF,s=28,marker='*',color='blue',label='Exp.')
#    plt.plot(t_ex,y_CBF, linestyle = '-.',color='steelblue')
    
    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 10})
    plt.xlabel('Time(h)',font)
    plt.ylabel('CBF3 mRNA level',font1)
    
#    file = pd.read_csv('heatmap_temp.csv', header = None, sep = ',') 
#
#    df = pd.DataFrame(file)
#
#    x1 = np.linspace(0,81,82)
#    yy2 = 6.95*np.ones(len(x1))
#    col = np.array(df[1])
#    plt.scatter(x1,yy2,s=80,marker='s',c=col,cmap='BuPu')
    plt.xlim([0,80.5])
    plt.ylim([0,7.1])
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    
    
    plt.subplot(133)
    plt.plot(t,y[9,:],color='black',label='Sim.') 
    y_COR = [0.130606347,0.106151288,0.1,0.117339906,0.122473311,0.158888556,0.206598895,0.744932524,1.949147799,1.370487294,1.33668181,1.337703334,1.452678684,1.788191671,2.09161031,2.4]
    plt.scatter(t_ex,y_COR,s=28,marker='o',color='blue',label='Exp.')

    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 10})
    plt.xlabel('Time(h)',font)
    plt.ylabel('COR15A mRNA level',font1)
    
#    file = pd.read_csv('heatmap_temp.csv', header = None, sep = ',') 
#
#    df = pd.DataFrame(file)
#
#    x1 = np.linspace(0,81,82)
#    yy2 = 6.95*np.ones(len(x1))
#    col = np.array(df[1])
#    sc1 = plt.scatter(x1,yy2,s=80,marker='s',c=col,cmap='BuPu')
    plt.xlim([0,80.5])
    plt.ylim([0,7.1])
#    plt.colorbar(sc1)
    
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    
    plt.show()
#    plt.plot(t_ex,y_COR, linestyle = '-.',color='steelblue')
#    plt.subplot(231)
#    plt.plot(t,y[0,:]**n0/(KCa**n0+y[0,:]**n0),'k',label='CaM')
#    plt.xlim([0,26.5])
#    plt.ylim([0,1.2])
#    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 10})
#    plt.xlabel('Time(h)',font)
#    plt.ylabel(r'CaM',font1)
#    plt.xticks(fontproperties = 'Times New Roman', size = 14)
#    plt.yticks(fontproperties = 'Times New Roman', size = 14)
#    x1 = np.linspace(0,2,2)
#    x2 = np.linspace(2,26,2)
#    yy1 = 1.14*np.ones(len(x1))
#    yy2 = 1.2*np.ones(len(x1))
#    plt.fill_between(x1,yy1,yy2,facecolor='red')
#    plt.fill_between(x2,yy1,yy2,facecolor='aliceblue')
#
#
#    plt.subplot(232)
##    plt.plot(t,y[2,:]/(max(y[2,:])),'k',label='ICE1')
##    plt.xlim([0,26.5])
##    plt.ylim([0,1.2])
##    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 10})
##    plt.xlabel('Time(h)',font)
##    plt.ylabel('ICE1(nM)',font1)
##    plt.xticks(fontproperties = 'Times New Roman', size = 14)
##    plt.yticks(fontproperties = 'Times New Roman', size = 14)
##    x1 = np.linspace(0,2,2)
##    x2 = np.linspace(2,26,2)
##    yy1 = 1.14*np.ones(len(x1))
##    yy2 = 1.2*np.ones(len(x1))
##    plt.fill_between(x1,yy1,yy2,facecolor='red')
##    plt.fill_between(x2,yy1,yy2,facecolor='aliceblue')
#    yy=y1[2,:]/(max(y1[2,:]))
#    yy[0:400]=0.4189801
#    plt.plot(t1,yy,label='ICE1',color='k')
#
#    plt.xlim([0,26])
#    plt.ylim([0,1.2])
#    plt.xlabel('Time(h)',font)
#    plt.ylabel('ICE1(nM)',font)
#    plt.xticks(fontproperties = 'Times New Roman', size = 14)
#    plt.yticks(fontproperties = 'Times New Roman', size = 14)
#    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 10})
#    x1 = np.linspace(0,2,2)
#    x2 = np.linspace(2,26,2)
#    yy1 = 1.14*np.ones(len(x1))
#    yy2 = 1.2*np.ones(len(x1))
#    plt.fill_between(x1,yy1,yy2,facecolor='red')
#    plt.fill_between(x2,yy1,yy2,facecolor='aliceblue')
# 
#
#    plt.subplot(233)
#    plt.plot(t,y[1,:]/2,'k',label='MYB15')
#    plt.xlim([0,26.5])
#    plt.ylim([0,1.2])
#    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 10})
#    plt.xlabel('Time(h)',font)
#    plt.ylabel('Fraction of MYB15',font1)
#    plt.xticks(fontproperties = 'Times New Roman', size = 14)
#    plt.yticks(fontproperties = 'Times New Roman', size = 14)
#    x1 = np.linspace(0,2,2)
#    x2 = np.linspace(2,26,2)
#    yy1 = 1.14*np.ones(len(x1))
#    yy2 = 1.2*np.ones(len(x1))
#    plt.fill_between(x1,yy1,yy2,facecolor='red')
#    plt.fill_between(x2,yy1,yy2,facecolor='aliceblue')
#
#
#    plt.subplot(234)
#    plt.plot(t,y[4,:],'k',label='CBF3 mRNA')
#    plt.xlim([0,26.5])
#    plt.ylim([0,4])
#    t_CBF3 = [2,3,5,8,11,26]
#    y_CBF3 = [0,2.041841004,2.661087866,1.087866109,0.20083682,0.066945607]
#    plt.scatter(t_CBF3,y_CBF3,s=12,color='r',label='Exp.')
#    plt.plot(t_CBF3,y_CBF3, linestyle = '-.',color='chocolate')
#    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 10})
#    plt.xlabel('Time(h)',font)
#    plt.ylabel('CBF3 mRNA(nM)',font1)
#    plt.xticks(fontproperties = 'Times New Roman', size = 14)
#    plt.yticks(fontproperties = 'Times New Roman', size = 14)
#    x1 = np.linspace(0,2,2)
#    x2 = np.linspace(2,26,2)
#    yy1 = 3.8*np.ones(len(x1))
#    yy2 = 4*np.ones(len(x1))
#    plt.fill_between(x1,yy1,yy2,facecolor='red')
#    plt.fill_between(x2,yy1,yy2,facecolor='aliceblue')
#
#    
#    plt.subplot(235)
#    plt.plot(t,y[7,:],'k',label='ZAT12 mRNA')
#    plt.xlim([0,26.5])
#    plt.ylim([0,4])
#    t_ZAT12 = [2,3,4,5,10,26]
#    y_ZAT12 = [0,1.196721311,1.360655738,2.885245902,2.081967213,0.967213115]
#    plt.scatter(t_ZAT12,y_ZAT12,s=12, color='b',label='Exp.')
#    plt.plot(t_ZAT12,y_ZAT12, linestyle = '-.',color='steelblue')
#    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 10})
#    plt.xlabel('Time(h)',font)
#    plt.ylabel('ZAT12 mRNA(nM)',font1)
#    plt.xticks(fontproperties = 'Times New Roman', size = 14)
#    plt.yticks(fontproperties = 'Times New Roman', size = 14)  
#    x1 = np.linspace(0,2,2)
#    x2 = np.linspace(2,26,2)
#    yy1 = 3.8*np.ones(len(x1))
#    yy2 = 4*np.ones(len(x1))
#    plt.fill_between(x1,yy1,yy2,facecolor='red')
#    plt.fill_between(x2,yy1,yy2,facecolor='aliceblue')
#       
#    plt.subplot(236)
#    plt.plot(t,y[9,:],'k',label='COR15A mRNA')
#    plt.xlim([0,26.5])
#    plt.ylim([0,4])
#    t_COR15A = [2.25,2.5,3,4,5,14,26]
#    y_COR15A = [0.08,0.2,0.379136063,0.789598263,1.108685406,2.389680438,2.74]
#    plt.scatter(t_COR15A,y_COR15A,s=12,color='g',label='Exp.')
#    plt.plot(t_COR15A,y_COR15A, linestyle = '-.',color='darkgreen')
#    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 10})
#    plt.xlabel('Time(h)',font)
#    plt.ylabel('COR15A mRNA(nM)',font1)
#    plt.xticks(fontproperties = 'Times New Roman', size = 14)
#    plt.yticks(fontproperties = 'Times New Roman', size = 14)
#    x1 = np.linspace(0,2,2)
#    x2 = np.linspace(2,26,2)
#    yy1 = 3.8*np.ones(len(x1))
#    yy2 = 4*np.ones(len(x1))
#    plt.fill_between(x1,yy1,yy2,facecolor='red')
#    plt.fill_between(x2,yy1,yy2,facecolor='aliceblue')
    
#    peak_id,peak_property = find_peaks(y[4,:], height=0, distance=24)
#    peak_time = t[peak_id]
#    print(peak_time)

    plt.show()