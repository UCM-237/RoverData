#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  7 15:53:30 2025

@author: lia
"""

import paparazzi_data as ppdat
import numpy as np
import matplotlib.pyplot as plt


#file="data202507/25_07_21__12_01_50.data"
#ppdat.split_rover_data(file)

'''Anibal'''
#ppdat.extract_data("data202507/21_12_01/anibal.dat")

datos_CBF=np.loadtxt("datos_nei.dat")
t,Xi,Xi_CBF,nnei,act_cond,d,R,alpha=ppdat.extract_CBF(datos_CBF)

l=len(t)
n_outliers=0
for i in range(l):
    for j in range(4):
        if d[i,j]>400 :
            d[i,j]=d[i-1,j]
            n_outliers=n_outliers+1


datos1=np.zeros((l,14))
datos1[:,0]=t
datos1[:,1:3]=Xi
datos1[:,3:5]=Xi_CBF
datos1[:,5]=nnei
datos1[:,6]=act_cond
datos1[:,7:11]=d
datos1[:,12]=R
datos1[:,13]=alpha

'''







plt.figure()
plt.plot(t,Xi[:,0],'b.',label="Xi_x")
plt.plot(t,Xi[:,1],'c.',label="Xi_y")
plt.plot(t,Xi_CBF[:,0],'r.',label="Xi_CBF_x")
plt.plot(t,Xi_CBF[:,1],'m.',label="Xi_CBF_y")
plt.legend()
plt.xlabel("t(s)")
plt.grid()

p=np.ones((l,2))
p[:,0]=t
plt.figure()
plt.quiver(p[:,0],p[:,1],Xi[:,0],Xi[:,1],color="red")
plt.quiver(p[:,0],p[:,1],Xi_CBF[:,0],Xi_CBF[:,1],color="blue")

plt.figure()
plt.subplot(2,1,1)
plt.axes(ylim=[0,10])
plt.plot(t,d)
plt.xlabel("t(s)")
plt.ylabel("d(m)")



plt.subplot(2,1,2)
plt.plot(t,act_cond)
plt.xlabel("t(s)")
plt.ylabel("Active conditions")

'''


gps=np.loadtxt("datos_gps.dat")
t_gps,x_gps,y_gps,z_gps,course,speed,utm_zone=ppdat.extract_utm_data(gps)
utm_t=[]
utm_x=[]
utm_y=[]

for i in range(len(t_gps)):
    if x_gps[i]!=0 and y_gps[i]!=0:
        utm_t.append(t_gps[i])
        utm_y.append(x_gps[i])
        utm_x.append(y_gps[i])
l2=len(utm_t)
datos2=np.zeros((l2,3))
datos2[:,0]=utm_t
datos2[:,1]=utm_x
datos2[:,2]=utm_y

t,out=ppdat.combine_generic(datos1, datos2, mode="auto")

plt.figure()
plt.quiver(out[:,14],out[:,13],out[:,0],out[:,1],color="red")
plt.quiver(out[:,14],out[:,13],out[:,2],out[:,3],color="blue")
plt.plot(out[:,14],out[:,13],'c-')

'''Avelino'''
#ppdat.extract_data("data202507/21_12_01/avelino.dat")




'''Pepa'''
#ppdat.extract_data("data202507/21_12_01/pepa.dat")
