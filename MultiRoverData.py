#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  7 15:53:30 2025

@author: lia
"""

import paparazzi_data as ppdat
import numpy as np
import matplotlib.pyplot as plt


file="data202507/25_07_21__12_01_50.data"
#ppdat.split_rover_data(file)

'''Anibal'''
ppdat.extract_data("data202507/21_12_01/anibal.dat")
nei_data=np.loadtxt("datos_nei.dat")

#datos_anibal=ppdat.extract_nei_data(nei_data, 6)
datos_avelino=ppdat.extract_nei_data(nei_data,4)
datos_pepa=ppdat.extract_nei_data(nei_data,5)
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

t_av=datos_avelino[0]
x_av=datos_avelino[1]
y_av=datos_avelino[2]
vx_av=datos_avelino[3]
vy_av=datos_avelino[4]
t_pp=datos_pepa[0]
x_pp=datos_pepa[1]
y_pp=datos_pepa[2]
vx_pp=datos_pepa[3]
vy_pp=datos_pepa[4]
plt.figure()

plt.plot(x_av,y_av,'m.-',label="Avelino")
plt.plot(x_pp,y_pp,'g.-',label="Pepa")
#plt.plot(utm_x,utm_y,'c.-',label="Anibal")
plt.xlabel("UTM-x (m)")
plt.ylabel("UTM-y (m)")
plt.title('Posiciones recibidas por Anibal')
plt.legend()
    
plt.figure()
plt.plot(t_av,x_av,'m.-',label="Avelino")
plt.plot(t_pp,x_pp,'g.-',label="Pepa")
#plt.plot(utm_t,utm_x,'c.-',label="Anibal")
plt.xlabel("t(s)")
plt.ylabel("UTM-x(m)")
plt.title('Posiciones recibidas por Anibal')
plt.legend()

plt.figure()
plt.plot(t_av,y_av,'m.-',label="Avelino")
plt.plot(t_pp,y_pp,'g.-',label="Pepa")
#plt.plot(utm_t,utm_y,'c.-',label="Tana")
plt.xlabel("t(s)")
plt.ylabel("UTM-y(m)")
plt.title('Posiciones recibidas por Anibal')
plt.legend()



'''Avelino'''
ppdat.extract_data("data202507/21_12_01/avelino.dat")
nei_data=np.loadtxt("datos_nei.dat")

datos_anibal=ppdat.extract_nei_data(nei_data, 6)
datos_pepa=ppdat.extract_nei_data(nei_data,5)

t_an=datos_anibal[0]
x_an=datos_anibal[1]
y_an=datos_anibal[2]
vx_an=datos_anibal[3]
vy_an=datos_anibal[4]
t_pp=datos_pepa[0]
x_pp=datos_pepa[1]
y_pp=datos_pepa[2]
vx_pp=datos_pepa[3]
vy_pp=datos_pepa[4]

plt.figure()
plt.plot(x_an,y_an,'b-',data="Anibal")
plt.plot(x_pp,y_pp,'g-',label="Pepa")
plt.title("Posiciones recibidas por Avelino")
plt.legend()

plt.figure()
plt.plot(t_an,x_an,'b-',data="Anibal")
plt.plot(t_pp,x_pp,'g-',label="Pepa")
plt.title("Posiciones recibidas por Avelino")
plt.legend()



'''Pepa'''
ppdat.extract_data("data202507/21_12_01/pepa.dat")
nei_data=np.loadtxt("datos_nei.dat")

datos_avelino=ppdat.extract_nei_data(nei_data, 4)
datos_anibal=ppdat.extract_nei_data(nei_data, 6)

gps=np.loadtxt("datos_gps.dat")
t_gps,x_gps,y_gps,z_gps,course,speed=ppdat.extract_gps_data3(gps)
utm_t=[]
utm_x=[]
utm_y=[]
for i in range(len(t_gps)):
    if x_gps[i]!=0 and y_gps[i]!=0:
        utm_t.append(t_gps[i])
        utm_y.append(x_gps[i]*(0.01))
        utm_x.append(y_gps[i]*0.01)

t_an=datos_anibal[0]
x_an=datos_anibal[1]
y_an=datos_anibal[2]
vx_an=datos_anibal[3]
vy_an=datos_anibal[4]
t_av=datos_avelino[0]
x_av=datos_avelino[1]
y_av=datos_avelino[2]
vx_av=datos_avelino[3]
vy_av=datos_avelino[4]
plt.figure()
plt.plot(x_av,y_av,'m.-',label="Avelino")
plt.plot(x_an,y_an,'b.-',label="Anibal")
plt.plot(utm_x,utm_y,'g.-',label="Pepa")
plt.title("Posiciones recibidas por Pepa")
plt.legend()

plt.figure()
plt.plot(t_av,x_av,'m.-',label="Avelino")
plt.plot(t_an,x_an,'b.-',label="Anibal")
#plt.plot(utm_x,utm_y,'g.-',label="Pepa")
plt.title("Posiciones recibidas por Pepa")
plt.legend()
