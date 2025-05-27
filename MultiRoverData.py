#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  7 15:53:30 2025

@author: lia
"""

import paparazzi_data as ppdat
import numpy as np
import matplotlib.pyplot as plt


file="data20250527/25_05_27__10_39_38.data"
ppdat.split_rover_data(file)

'''Tana'''
ppdat.extract_data("data20250527/tana.dat")
nei_data=np.loadtxt("datos_nei.dat")

datos_anibal=ppdat.extract_nei_data(nei_data, 6)
datos_avelino=ppdat.extract_nei_data(nei_data,4)
datos_pepa=ppdat.extract_nei_data(nei_data,5)
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
x_an=t_an=datos_anibal[1]
y_an=datos_anibal[2]
vx_an=datos_anibal[3]
vy_an=datos_anibal[4]
t_av=datos_avelino[0]
x_av=t_an=datos_avelino[1]
y_av=datos_avelino[2]
vx_av=datos_avelino[3]
vy_av=datos_avelino[4]
t_pp=datos_pepa[0]
x_pp=t_an=datos_pepa[1]
y_pp=datos_pepa[2]
vx_pp=datos_pepa[3]
vy_pp=datos_pepa[4]
plt.figure()
plt.plot(x_an,y_an,'b.-',label="Anibal")
plt.plot(x_av,y_av,'m.-',label="Avelino")
plt.plot(x_pp,y_pp,'g.-',label="Pepa")
plt.plot(utm_x,utm_y,'c.-',label="Tana")
plt.title('Posiciones recibidas por Tana')
plt.legend()

'''Avelino
ppdat.extract_data("data20250527/avelino.dat")
nei_data=np.loadtxt("datos_nei.dat")

datos_anibal=ppdat.extract_nei_data(nei_data, 6)
datos_tana=ppdat.extract_nei_data(nei_data,1)
datos_pepa=ppdat.extract_nei_data(nei_data,5)

t_an=datos_anibal[0]
x_an=t_an=datos_anibal[1]
y_an=datos_anibal[2]
vx_an=datos_anibal[3]
vy_an=datos_anibal[4]
t_ta=datos_tana[0]
x_ta=t_an=datos_tana[1]
y_ta=datos_tana[2]
vx_ta=datos_tana[3]
vy_ta=datos_tana[4]
t_pp=datos_pepa[0]
x_pp=t_an=datos_pepa[1]
y_pp=datos_pepa[2]
vx_pp=datos_pepa[3]
vy_pp=datos_pepa[4]

plt.figure()
plt.plot(x_an,y_an,'b-',data="Anibal")
plt.plot(x_pp,y_pp,'g-',label="Pepa")
plt.plot(x_pp,y_pp,'c-',label="Tana")
plt.title("Posiciones recibidas por Avelino")
plt.legend()

'''
'''Anibal'''
ppdat.extract_data("data20250527/anibal.dat")
nei_data=np.loadtxt("datos_nei.dat")

datos_avelino=ppdat.extract_nei_data(nei_data, 4)
datos_tana=ppdat.extract_nei_data(nei_data,1)
datos_pepa=ppdat.extract_nei_data(nei_data,5)
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

t_av=datos_avelino[0]
x_av=t_an=datos_avelino[1]
y_av=datos_avelino[2]
vx_av=datos_avelino[3]
vy_av=datos_avelino[4]
t_ta=datos_tana[0]
x_ta=t_an=datos_tana[1]
y_ta=datos_tana[2]
vx_ta=datos_tana[3]
vy_ta=datos_tana[4]
t_pp=datos_pepa[0]
x_pp=t_an=datos_pepa[1]
y_pp=datos_pepa[2]
vx_pp=datos_pepa[3]
vy_pp=datos_pepa[4]


plt.figure()
plt.plot(x_av,y_av,'m.-',label="Avelino")
plt.plot(x_pp,y_pp,'g.-',label="Pepa")
plt.plot(x_ta,y_ta,'c.-',label="Tana")
plt.plot(utm_x,utm_y,'b.-',label="Anibal")
plt.title("Posiciones recibidas por Anibal")
plt.legend()




'''Pepa'''
ppdat.extract_data("data20250527/pepa.dat")
nei_data=np.loadtxt("datos_nei.dat")

datos_avelino=ppdat.extract_nei_data(nei_data, 4)
datos_anibal=ppdat.extract_nei_data(nei_data, 6)
datos_tana=ppdat.extract_nei_data(nei_data,1)
'''
fname="datos_ll.dat"
datos_lonlat=np.loadtxt(fname)
t_ll,lat,lon=ppdat.extract_gps_ll(datos_lonlat)
ll=len(lon)
geod=np.zeros((ll,2))
geod[:,0]=lon[:]
geod[:,1]=lat[:]



utm = np.array([transformer.transform(lon, lat) for lon, lat in geod])
'''
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
x_an=t_an=datos_anibal[1]
y_an=datos_anibal[2]
vx_an=datos_anibal[3]
vy_an=datos_anibal[4]
t_av=datos_avelino[0]
x_av=t_an=datos_avelino[1]
y_av=datos_avelino[2]
vx_av=datos_avelino[3]
vy_av=datos_avelino[4]
t_ta=datos_tana[0]
x_ta=t_an=datos_tana[1]
y_ta=datos_tana[2]
vx_ta=datos_tana[3]
vy_ta=datos_tana[4]
plt.figure()
plt.plot(x_av,y_av,'m.-',label="Avelino")
plt.plot(x_ta,y_ta,'c.-',label="Tana")
plt.plot(x_an,y_an,'b.-',label="Anibal")
plt.plot(utm_x,utm_y,'g.-',label="Pepa")
plt.title("Posiciones recibidas por Pepa")
plt.legend()
