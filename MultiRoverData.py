#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  7 15:53:30 2025

@author: lia
"""

import paparazzi_data as ppdat
import numpy as np
import matplotlib.pyplot as plt


file="data20250507/25_05_07__12_32_39.data"
#ppdat.split_rover_data(file)
'''Tana'''
ppdat.extract_data("tana.dat")
nei_data=np.loadtxt("datos_nei.dat")

datos_anibal=ppdat.extract_nei_data(nei_data, 6)
datos_avelino=ppdat.extract_nei_data(nei_data,4)


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
plt.figure()
plt.plot(x_an,y_an,'b-',label="Anibal")
plt.plot(x_av,y_av,'m-',label="Avelino")
plt.title('Posiciones recibidas por Tana')
plt.legend()

'''Avelino'''
ppdat.extract_data("data20250507/avelino.dat")
nei_data=np.loadtxt("datos_nei.dat")

datos_anibal=ppdat.extract_nei_data(nei_data, 6)
datos_tana=ppdat.extract_nei_data(nei_data,1)

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
plt.figure()
plt.plot(x_an,y_an,'b-',data="Anibal")
plt.title("Posiciones recibidas por Avelino")
plt.legend()


'''Anibal'''
ppdat.extract_data("data20250507/anibal.dat")
nei_data=np.loadtxt("datos_nei.dat")

datos_avelino=ppdat.extract_nei_data(nei_data, 4)
datos_tana=ppdat.extract_nei_data(nei_data,1)

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
plt.plot(x_av,y_av,'b-',data="Avelino")
#plt.plot(x_ta,y_ta,'m-',data="Tana")
plt.title("Posiciones recibidas por Anibal")
plt.legend()



