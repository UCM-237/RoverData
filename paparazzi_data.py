#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:14:41 2024
Module to analyse paparazzy data
@author: lia
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon 
import matplotlib.colors as mcolors
from scipy.signal import butter, lfilter, freqz, kaiserord, firwin, filtfilt

def split_rover_data(file):
    tana=""
    avelino=""
    anibal=""
    with open(file) as archivo:
        for linea in archivo:
            x=linea.split() 
            if x[1]=="1":
                tana=tana+linea
            elif x[1]=="4":
                avelino=avelino+linea
            elif x[1]=="6":
                anibal=anibal+linea
    archivo.close()
    with open("tana.dat","w") as archivo: 
        archivo.write(tana)
    archivo.close()
    
    with open("avelino.dat","w") as archivo:
        archivo.write(avelino)
    archivo.close()
    
    with open("anibal.dat","w") as archivo:
        archivo.write(anibal)
    archivo.close()
    return 0
        
def extract_data(file):
    sonar_data = ""
    gps_data = "" 
    inertial_data = "" 
    rc_data = ""
    actuator_data = ""
    energy_data = ""
    compass_data=""
    gvf_data=""
    link_data=""
    static_data=""
    control_data=""
    nei_data=""
    with open(file) as archivo:
        for linea in archivo:
            if "BR_SONAR" in linea:
                x=linea.split("BR_SONAR")  
                sonar_data +=x[0]+x[1]
            elif "GPS" in linea:
               x=linea.split("GPS")
               if "GPS_INT" in linea:
                  continue
               elif "GPS_SOL" in linea:
                   continue
               gps_data +=x[0]+x[1]
            elif "INS" in linea:
               if "INS_REF" in linea:
                  continue
               x=linea.split("INS")
               inertial_data +=x[0]+x[1]         
            elif "RC " in linea:
               x=linea.split("RC")
               rc_data+=x[0]+x[1]
            elif "ACTUATORS" in linea:
                x=linea.split("ACTUATORS")
                #x=y.split(",")
                actuator_data+=x[0]+x[1]
            elif "ENERGY" in linea:
               x=linea.split("ENERGY")
               energy_data+=x[0]+x[1]
            elif "IMU_MAG_RAW" in linea:
                  x=linea.split("IMU_MAG_RAW")
                  compass_data+=x[0]+x[1]
            elif "GVF_PARAMETRIC" in linea:
                x=linea.split("GVF_PARAMETRIC")
                gvf_data+=x[0]+x[1]
            elif "DATALINK_REPORT" in linea:
                x=linea.split("DATALINK_REPORT")
            elif "LINK_REPORT" in linea: 
                x=linea.split("LINK_REPORT")
                link_data+=x[0]+x[1]
            elif "STATIC_CONTROL" in linea:
                x=linea.split("STATIC_CONTROL")
                static_data+=x[0]+x[1]
            elif "BOAT_CTRL" in linea:
                x=linea.split("BOAT_CTRL")
                control_data+=x[0]+x[1]
            elif "CBF_REC" in linea:
                x=linea.split("CBF_REC")
                nei_data+=x[0]+x[1]
                
    archivo.close()
    with open("datos_sonar.dat","w") as archivo: 
        archivo.write(sonar_data) 
    archivo.close()
    with open("datos_gps.dat","w") as archivo:
        archivo.write(gps_data)
    archivo.close()
   
    with open("datos_ins.dat","w") as archivo:
        archivo.write(inertial_data)
    archivo.close()
   
    with open("datos_rc.dat","w") as archivo:
        archivo.write(rc_data)
    archivo.close()
   
    with open("datos_act.dat","w") as archivo:
        archivo.write(actuator_data)
    archivo.close()

    with open("datos_energy.dat","w") as archivo:
        archivo.write(energy_data)
    archivo.close()
    
    with open("datos_compass.dat","w") as archivo:
        archivo.write(compass_data)
    archivo.close()
    
    with open("datos_gvf.dat","w") as archivo:
        archivo.write(gvf_data)
    archivo.close()
    with open("datos_link.dat","w") as archivo:
        archivo.write(link_data)
    archivo.close()
    
    with open("datos_static.dat","w") as archivo:
         archivo.write(static_data)
    archivo.close()

    with open("datos_control.dat","w") as archivo:
         archivo.write(control_data)
    archivo.close()

    with open("datos_nei.dat","w") as archivo:
         archivo.write(nei_data)
    archivo.close()

    return 0    

def low_pass_filter(x, samples = 20):
  #fft based brute force low pass filter 
    a = np.fft.rfft(x) 
    tot = len(a)
    for x in range(tot-samples):
        a[samples + x] = 0.0
    return np.fft.irfft(a)

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def extract_gps_data(datos_gps):
    f,c =np.shape(datos_gps)
    x_gps=np.zeros(f)
    y_gps=np.zeros(f)
    z_gps=np.zeros(f)
    t_gps=np.zeros(f)
    xhome=datos_gps[0,3]
    yhome=datos_gps[0,4]
    zhome=datos_gps[0,5]
    t_gps[:]=datos_gps[:,0]
    x_gps[:]=datos_gps[:,3]-xhome
    y_gps[:]=datos_gps[:,4]-yhome
    z_gps[:]=datos_gps[:,5]-zhome
    return t_gps,x_gps,y_gps,z_gps
'''
<message name="GPS" id="8">
<field name="mode" type="uint8" unit="byte_mask"/>
<field name="utm_east" type="int32" unit="cm" alt_unit="m"/>
<field name="utm_north" type="int32" unit="cm" alt_unit="m"/>
<field name="course" type="int16" unit="decideg" alt_unit="deg"/>
<field name="alt" type="int32" unit="mm" alt_unit="m">Altitude above geoid (MSL)</field>
<field name="speed" type="uint16" unit="cm/s" alt_unit="m/s">norm of 2d ground speed in cm/s</field>
<field name="climb" type="int16" unit="cm/s" alt_unit="m/s"/>
<field name="week" type="uint16" unit="weeks"/>
<field name="itow" type="uint32" unit="ms"/>
<field name="utm_zone" type="uint8"/>
<field name="gps_nb_err" type="uint8"/>
</message>
'''

def extract_gps_data2(datos_gps):
    f,c =np.shape(datos_gps)
    x_gps=np.zeros(f)
    y_gps=np.zeros(f)
    z_gps=np.zeros(f)
    course=np.zeros(f)
    speed=np.zeros(f)
    t_gps=np.zeros(f)
    xhome=datos_gps[0,3]
    yhome=datos_gps[0,4]
    zhome=datos_gps[0,6]
    t_gps[:]=datos_gps[:,0]
    x_gps[:]=datos_gps[:,3]-xhome
    y_gps[:]=datos_gps[:,4]-yhome
    z_gps[:]=datos_gps[:,6]-zhome
    course[:]=datos_gps[:,5]
    speed[:]=datos_gps[:,7]
    return t_gps,x_gps,y_gps,z_gps,course,speed

def extract_act_data(datos_actuadores):
    f,c =np.shape(datos_actuadores)
    #print(f,c)
    t_act=np.zeros(f)
    md=np.zeros(f)
    mi=np.zeros(f)
    t_act[:]=datos_actuadores[:,0]
    md[:]=datos_actuadores[:,2]
    mi[:]=datos_actuadores[:,3]
    return t_act,md,mi

def extract_compass_data(datos_brujula):
    f,c =np.shape(datos_brujula)
    #print(f,c)
    t_com=np.zeros(f)
    ang_x=np.zeros(f)
    ang_y=np.zeros(f)
    ang_z=np.zeros(f)
    t_com[:]=datos_brujula[:,0]
    ang_x[:]=datos_brujula[:,3]
    ang_y[:]=datos_brujula[:,4]
    ang_z[:]=datos_brujula[:,5]

    return t_com,ang_x,ang_y,ang_z

def extract_ins_data(datos_inercial):
    f,c =np.shape(datos_inercial)
    
    t_inercial=np.zeros(f)
    x=np.zeros(f)
    y=np.zeros(f)
    z=np.zeros(f)
    xd=np.zeros(f)
    yd=np.zeros(f)
    zd=np.zeros(f)
    xdd=np.zeros(f)
    ydd=np.zeros(f)
    zdd=np.zeros(f)
    t_inercial[:]=datos_inercial[:,0]
    pos_fac=0.0039063
    vel_fac=0.0000019
    ace_fac=0.0009766
    x[:]=datos_inercial[:,2]*pos_fac
    y[:]=datos_inercial[:,3]*pos_fac
    z[:]=datos_inercial[:,4]*pos_fac
    xd[:]=datos_inercial[:,5]*vel_fac
    yd[:]=datos_inercial[:,6]*vel_fac
    zd[:]=datos_inercial[:,7]*vel_fac
    xdd[:]=datos_inercial[:,8]*ace_fac
    ydd[:]=datos_inercial[:,9]*ace_fac
    zdd[:]=datos_inercial[:,10]*ace_fac
    return t_inercial,x,y,z,xd,yd,zd,xdd,ydd,zdd

def extract_gvf_data(datos_gvf):
    f,c =np.shape(datos_gvf)
    w=np.zeros(f)
    s=np.zeros(f)
    p=np.zeros([f,15])
    t=np.zeros(f)
    fi1=np.zeros(f)
    fi2=np.zeros(f)
    t[:]=datos_gvf[:,0]
    s[:]=datos_gvf[:,3]
    w[:]=datos_gvf[:,4]
    p[:,:]=datos_gvf[:,5:20]
    fi1[:]=datos_gvf[:,21]
    fi2[:]=datos_gvf[:,22]

    return t,w,s,p,fi1,fi2

def extract_link_data(datos_link):
    f,c =np.shape(datos_link)
    lost_time=np.zeros(f)
    t=np.zeros(f)
    t[:]=datos_link[:,0]
    lost_time[:]=datos_link[:,4]
   
    return t,lost_time

def extract_nei_data(nei_data,id_nei):
    f,c =np.shape(nei_data)
    x=[]
    y=[]
    vx=[]
    vy=[]
    t=[]
    for i in range(f):
        if(nei_data[i,2]==id_nei):
            t.append(nei_data[i,0])
            x.append(nei_data[i,5])
            y.append(nei_data[i,6])
            vx.append(nei_data[i,7])
            vy.append(nei_data[i,8])
    return t,x,y,vx,vy
    
    
def plot_gps(x_gps,y_gps):
    fig,ax= plt.subplots(1,1)
    ax.plot(x_gps,y_gps,'b.')
    ax.set_aspect("equal")
    plt.xlabel("x (cm)")
    plt.ylabel("y (cm)")
    plt.grid()
    plt.show()
    return

def plot_gps_tramos(x_gps,y_gps,indices):
    n=len(indices)
    plt.figure()
   
    ax1=plt.subplot(n-1,1,1)
    for i in range(n-1):
        i_i=indices[i]
        i_f=indices[i+1]
        plt.subplot(n-1,1,i+1,sharex=ax1,sharey=ax1)
        plt.plot(x_gps[i_i:i_f],y_gps[i_i:i_f],'b.')
        #plt.title("Trajectory")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.grid(visible=True)
    plt.show()
    return

def plot_ins(t_inercial,xd,xdf,yd,ydf):
    plt.figure()
    plt.plot(t_inercial,xd,'g',label= 'datos en crudo')
    plt.plot(t_inercial,xdf,'b', label='datos filtrados')
    plt.title("Velocidades unidad inercial")
    plt.legend(loc =2)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Velocidad eje x (m/s)")
    plt.figure()
    plt.plot(t_inercial,yd,'g',label= 'datos en crudo')
    plt.plot(t_inercial,ydf,'b', label='datos filtrados')
    plt.title("Velocidades unidad inercial")
    plt.legend(loc =2)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Velocidad eje y (m/s)")
    return

def plot_act(t_act,md,mi):
    plt.figure()
    plt.plot(t_act,md,'b-')
    plt.plot(t_act,mi,'g-')
    plt.title("Datos de los actuadores")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Actuador")
    plt.grid()
    plt.show()
    
def get_vel(ini,fin1,t,x,y,md,mi):

    A = np.vstack([t[ini:fin1].T, np.ones(len(t[ini:fin1]))]).T
    m, c = np.linalg.lstsq(A, x[ini:fin1], rcond=None)[0]
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(t[ini:fin1],x[ini:fin1],'b.')
    plt.xlabel("t(s)")
    plt.ylabel("x(m)")
    tp=np.linspace(t[ini],t[fin1],100)
    xp=m*tp+c
    plt.plot(tp,xp,'r-')
    plt.grid()
    v1x=m
    m, c = np.linalg.lstsq(A, y[ini:fin1], rcond=None)[0]
    plt.subplot(2,2,2)
    plt.plot(t[ini:fin1],y[ini:fin1],'b.')
    yp=m*tp+c
    plt.plot(tp,yp,'r-')
    plt.xlabel("t(s)")
    plt.ylabel("y(m)")
    plt.grid()
    v1y=m
    print("Velocidad=( ",v1x,"\t",v1y,")")
    
    
    plt.subplot(2,2,3)
    thr=0.5*(md[ini:fin1]+mi[ini:fin1])
    plt.plot(t[ini:fin1],0.5*(md[ini:fin1]+mi[ini:fin1]))
    plt.xlabel('t(s)')
    plt.ylabel('Throttle')
    plt.subplot(2,2,4)
    par=(md[ini:fin1]-mi[ini:fin1])
    plt.plot(t[ini:fin1],(md[ini:fin1]-mi[ini:fin1]))
    plt.xlabel('t(s)')
    plt.ylabel('Par')
    return v1x,v1y,np.mean(par),np.mean(thr)

def get_v(ini,fin,xf,yf,xdf,ydf,xddf,yddf,t):
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(t[ini:fin],xdf[ini:fin])
    plt.grid()
    plt.xlabel("t(s)")
    plt.ylabel("vx(m/s)")

    plt.subplot(2,1,2)
    plt.plot(t[ini:fin],ydf[ini:fin])
    plt.grid()
    plt.xlabel("t(s)")
    plt.ylabel("vy(m/s)")
    return np.mean(xdf[ini:fin]),np.mean(ydf[ini:fin])

def plot_inercial(ini,fin,x,y,xd,vd):
    plt.figure()
    plt.plot(x[ini:fin],y[ini:fin],'b.',x[ini],y[ini],'mo',x[fin],y[fin],'m*')
    
    plt.xlabel('X(m)')
    plt.ylabel('Y(m)')
    plt.axis('equal')
    plt.grid()
    plt.show()

def ajuste(x,y):
    A = np.vstack([x.T, np.ones(len(x))]).T
    print(np.shape(A))
    print(np.shape(y))
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m,c

def get_extremes(data):
    M=np.max(data)
    m=np.min(data)
    return M,m

'''
0 West
90 West
180 South
270 East
'''
def map_compass_data(ory,orx):
    ang=np.arctan2(ory,orx)
    n=len(ang)
    '''
    ang=np.rad2deg(ang)
    n=len(ang)
    for i in range(n):
        if ang[i]<0:
            ang[i]+=360
    '''
    for i in range(n):
        if ang[i]<0:
            ang[i]+=2*np.pi
        if ang[i]>2*np.pi:
            ang[i]-=2*np.pi
    return ang

def orientation2north(angle):
    angle2=angle+np.pi*0.5
    n=len(angle2)
    for i in range(n):
        if angle2[i]>2*np.pi:
                angle2[i]-=2*np.pi
    return angle2

def get_triangle(xr,yr,thetar):
    l=0.05
    v=np.zeros([3,2])
    v[0,0]=xr+l*np.cos(thetar)
    v[0,1]=yr+l*np.sin(thetar)
    v[1,0]=xr-l*0.5*np.cos(thetar-np.pi/4)
    v[1,1]=yr-l*0.5*np.sin(thetar-np.pi/4)
    v[2,0]=xr+l*0.5*np.sin(thetar-np.pi/4)
    v[2,1]=yr-l*0.5*np.cos(thetar-np.pi/4)
    return v

def purgar_datos(t,t_purga,var):
    k=0
    n=len(t)
    var_p=np.zeros(n)
    t_p=np.zeros(n)
    for i in range(n):
        ti=t[i]
        while t_purga[k]<ti:
            k+=1
        var_p[i]=var[k]
        t_p[i]=t_purga[k]
    return t_p,var_p
        
def ajuste_circulo(x,y):
    #En facil
    # Asumo que el centro es el centro de gravedad de los puntos
    xc=0
    yc=0
    r=0
    xc=np.mean(x)
    yc=np.mean(y)
    r2=np.mean((x-xc)**2+(y-yc)**2)
    r=np.sqrt(r2)
    return xc,yc,r

def plot_circle_data(x,y,xc,yc,r):
    
    n=len(x)
    plt.plot(x,y,'b*',x[0],y[0],'mo',x[n-1],y[n-1],'m*')
    xdib=np.linspace(xc-r,xc+r,100)
    ydp=np.zeros(len(xdib))
    ydn=np.zeros(len(xdib))
    for i in range(len(xdib)):
        discriminante=r**2-(xdib[i]-xc)**2
        if discriminante<0:
            discriminante=0
        ydp[i]=yc+np.sqrt(discriminante)
        ydn[i]=yc-np.sqrt(discriminante)
    plt.plot(xdib,ydp,'g-',xdib,ydn,'g-',xc,yc,'gx')
    plt.grid()
    plt.axis("equal")
    # Plot vectors
    #for i in range(0,n,5):
     #   plt.quiver(x[i],y[i],xc-x[i],yc-y[i],color= 'g',width=0.002)

def get_normal_component(x,y,xc,yc,xd,yd):
    vn=np.array([xc-x,yc-y])
    vn_norm=vn/np.linalg.norm(vn)
    v_tan=np.array([-yc+y,xc-x])
    v_tan_norm=v_tan/np.linalg.norm(v_tan)
    proyec_normal=np.dot(vn_norm,[[xd],[yd]]).item()
    proyec_tangencial=np.dot(v_tan_norm,[[xd],[yd]]).item()
    return proyec_normal,proyec_tangencial

def get_index_at_time(t,t0,t1):
    i=np.zeros([2],dtype=int)
    for j in range(len(t)):
        if t[j]>t0 and i[0]==0:
            i[0]=j
            continue
        if t[j]>t1 and i[1]==0:
            i[1]=j
            break
    return i

def mux_compute(x,y,t_ins,xd,yd,xdd,ydd,t_comp,grados,t0,t1,plot_circle=True):
     
    i_comp=get_index_at_time(t_comp, t0, t1)
    i_ins=get_index_at_time(t_ins, t0, t1)
    print("i_ins: ",i_ins)

    xr=np.zeros(np.shape(x[i_ins[0]:i_ins[1]]))
    yr=np.zeros(np.shape(x[i_ins[0]:i_ins[1]]))

    xr=x[i_ins[0]:i_ins[1]]    
    yr=y[i_ins[0]:i_ins[1]]    
    
    n_ins=len(xr)
    dat_inercial=np.zeros([n_ins,7])
    dat_inercial[:,0]=t_ins[i_ins[0]:i_ins[1]]
    dat_inercial[:,1]=xr
    dat_inercial[:,2]=yr
    dat_inercial[:,3]=xd[i_ins[0]:i_ins[1]]    
    dat_inercial[:,4]=yd[i_ins[0]:i_ins[1]]    
    dat_inercial[:,5]=xdd[i_ins[0]:i_ins[1]]    
    dat_inercial[:,6]=ydd[i_ins[0]:i_ins[1]]    
   
    
    radianes=np.deg2rad(grados[i_comp[0]:i_comp[1]])
    radianes=orientation2north(radianes)
    
    
    
    
    n_comp=len(radianes)

    dat_comp=np.zeros([n_comp,2])
    dat_comp[:,0]=t_comp[i_comp[0]:i_comp[1]]
    dat_comp[:,1]=radianes
    t,pos=combine_data(dat_inercial,dat_comp)
    fig,ax= plt.subplots(1,1)
    ax.plot(pos[0,0],pos[0,1],'co')
    #ax.plot(pos[:,0],pos[:,1],'b.')

    ax.set_aspect("equal")
    n=len(t)    
    plt.title("Trajectory")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.grid()

    v_w=np.array([2,1])
    nor=np.zeros(n)
    tan=np.zeros(n)
    ac=np.zeros(n)
    xc,yc,r=ajuste_circulo(pos[:,0],pos[:,1])
    print("xc,yc ",xc,yc)
    if(plot_circle==True):
        ax.plot(xc,yc,'mx')
        plot_circle_data(pos[:,0],pos[:,1], xc, yc, r)
    #TODO Revisar que las flechitas son correctas
    else:
        ax.plot(pos[:,0],pos[:,1],'b-')
   
    for i in range(n):
        
        if np.mod(i,200)==0:
            ax.plot(pos[i,0],pos[i,1],'b.')
            vertices=get_triangle(pos[i,0],pos[i,1], pos[i,2])
            ax.add_patch(Polygon(vertices, closed=True,fill=True,facecolor=mcolors.CSS4_COLORS['violet'],edgecolor=mcolors.CSS4_COLORS['darkviolet']))
            coseno=np.cos(-pos[i,2]+0.5*np.pi)
            seno=np.sin(-pos[i,2]+0.5*np.pi)
            R=np.array([[coseno, -seno],[seno, coseno]])
            v_b=np.array(([pos[i,3],-pos[i,4]]))
            v_w=np.dot(R,v_b)
            #ax.quiver(pos[i,0],pos[i,1],v_w[0],v_w[1],color= 'g',width=0.002)
            ax.quiver(pos[i,0],pos[i,1],pos[i,3],pos[i,4],color='c',width=0.002)
        nor[i],tan[i]=get_normal_component(pos[i,0],pos[i,1],xc,yc,pos[i,3],pos[i,4])
        ac[i]=np.linalg.norm(v_b)**2/r
        #ac[i]=tan[i]**2/r

    plt.show()   

    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(t,nor,'m.')
    plt.grid()
    plt.title("Componente normal")
    plt.subplot(3,1,2)
    plt.plot(t,tan,'c.')
    plt.title("Componente tangencial")
    plt.grid()
    plt.subplot(3,1,3)
    plt.plot(t,ac,'g.')
    plt.ylabel("Aceleracion centripeta (m/s²)")
    plt.xlabel("t(s)")
    plt.grid()

    ''' Con la componente centripeta puedo calcular el mu_x
    como la proyección de la centrípeta sobre el eje x del barco (R)
    R=mu_x*pdot
    '''
    #TODO: Revisar este calculo
    normal=np.zeros(2)
    Rvec=np.zeros(2)
    mu_x=np.zeros(n)
    Rvec=np.zeros(n)

    for i in range(n):
        coseno=np.cos(radianes[i]-0.5*np.pi)
        seno=np.sin(radianes[i]-0.5*np.pi)
        R=np.array([[coseno, -seno],[seno, coseno]])
        R=np.transpose(R) #Paso de ejes mundo a cuerpo
        mupy=np.dot(R,nor[i])
        if np.abs(pos[i,4])>0.000001:
            mu_x[i]=np.linalg.norm(mupy)/np.abs(pos[i,4])
        else:
            mu_x[i]=0
    print(Rvec)  
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(t,mu_x,'m.')
    plt.title("mu_x")
    plt.grid()

    plt.subplot(3,1,2)
    plt.plot(t,Rvec,'m.')
    plt.title("Proyeccion")
    plt.grid()

    plt.subplot(3,1,3)
    plt.plot(t,pos[:,4],'m.')
    plt.title("pdot")
    plt.grid()

    return np.mean(mu_x),r

def combine_data(dat_inercial,dat_comp):
    t_comp=dat_comp[:,0]
    grados=dat_comp[:,1]
    t_inercial=dat_inercial[:,0]
    xf=dat_inercial[:,1]
    yf=dat_inercial[:,2]
    xdf=dat_inercial[:,3]
    ydf=dat_inercial[:,4]
    xddf=dat_inercial[:,5]
    yddf=dat_inercial[:,6]
    n_comp=len(t_comp)
    n_ins=len(t_inercial)
    j=0
    pos=np.zeros([n_comp-1,7])
    t=np.zeros(n_comp-1)
    for i in range(n_comp-1):
        #print(t_inercial[j]-t_comp[i])
        if j>(n_ins-2):
            pos[i,0]=np.interp(t_comp[i],t_inercial[j-1:j], xf[j-1:j])
            pos[i,1]=np.interp(t_comp[i],t_inercial[j-1:j], yf[j-1:j])
            pos[i,2]=grados[i]
            pos[i,3]=np.interp(t_comp[i],t_inercial[j-1:j], xdf[j-1:j])
            pos[i,4]=np.interp(t_comp[i],t_inercial[j-1:j], ydf[j-1:j])
            pos[i,5]=np.interp(t_comp[i],t_inercial[j-1:j], xddf[j-1:j])
            pos[i,6]=np.interp(t_comp[i],t_inercial[j-1:j], yddf[j-1:j])
            t[i]=t_comp[i]
            continue
        if np.abs(t_inercial[j]-t_comp[i])<0.5:
            t[i]=t_comp[j]
            pos[i,0]=xf[j]
            pos[i,1]=yf[j]
            pos[i,2]=grados[i]
            pos[i,3]=xdf[j]
            pos[i,4]=ydf[j]
            pos[i,5]=xddf[j]
            pos[i,6]=yddf[j]
            #print("Coinciden:\t",t[i],"\t",pos[i])
            j+=1
            
        else:
            pos[i,0]=np.interp(t_comp[i],t_inercial[j-1:j], xf[j-1:j])
            pos[i,1]=np.interp(t_comp[i],t_inercial[j-1:j], yf[j-1:j])
            pos[i,2]=grados[i]
            pos[i,3]=np.interp(t_comp[i],t_inercial[j-1:j], xdf[j-1:j])
            pos[i,4]=np.interp(t_comp[i],t_inercial[j-1:j], ydf[j-1:j])
            pos[i,5]=np.interp(t_comp[i],t_inercial[j-1:j], xddf[j-1:j])
            pos[i,6]=np.interp(t_comp[i],t_inercial[j-1:j], yddf[j-1:j])
            t[i]=t_comp[i]
            #print("No coinciden:\t",t[i],"\t",pos[i])
    return t,pos


def ins_filtered(x,y,xd,yd,xdd,ydd):
    bw = 0.05
    ap = 20.0
    wc = 0.1
    N, beta = kaiserord(ap, bw)
    N += 1
    B = firwin(N, wc, window=('kaiser', beta))
    xf = filtfilt(B, 1.0, x)
    yf = filtfilt(B,1.0,y)
    
    wc = 0.25
    N, beta = kaiserord(ap, bw)
    N += 1
    B = firwin(N, wc, window=('kaiser', beta))
    xddf = filtfilt(B, 1.0, xdd)
    yddf = filtfilt(B,1.0,ydd)
   
    wc = 0.25
    N, beta = kaiserord(ap, bw)
    N += 1
    B = firwin(N, wc, window=('kaiser', beta))
    xdf = filtfilt(B, 1.0, xd)
    ydf = filtfilt(B,1.0,yd)
    return xf,yf,xdf,ydf,xddf,yddf

def compass_filtered(orx,ory):
    bw = 0.05
    ap = 20.0
    wc = 0.1
    N, beta = kaiserord(ap, bw)
    N += 1
    B = firwin(N, wc, window=('kaiser', beta))
    orxf = filtfilt(B, 1.0, orx)
    oryf = filtfilt(B,1.0,ory)
    return orxf,oryf

def extract_energy_data(datos_energia):
    f,c =np.shape(datos_energia)
    #print(f,c)
    t_eng=np.zeros(f)
    pwm=np.zeros(f)
    v=np.zeros(f)
    i=np.zeros(f)
    p=np.zeros(f)
    t_eng[:]=datos_energia[:,0]
    pwm[:]=datos_energia[:,2]
    v[:]=datos_energia[:,3]
    i[:]=datos_energia[:,4]
    p[:]=datos_energia[:,4]
    return t_eng,pwm,v,i,p

def extract_sonar_data(datos_sonar):
    f,c =np.shape(datos_sonar)
    #print(f,c)
    t_eng=np.zeros(f)
    d=np.zeros(f)
    conf=np.zeros(f)
    t_eng[:]=datos_sonar[:,0]
    d[:]=datos_sonar[:,12]
    conf[:]=datos_sonar[:,13]
    return t_eng,d,conf

def extract_static_control(datos_static):
    '''<message name="STATIC_CONTROL" id="158">
    <field name="active" type="int16" values="INACTIVE|ACTIVE"/>
    <field name="dist_WP" type="float"/>
    <field name="pt_bz" type="uint8"/>
    <field name="pxd" type="float"/>
    <field name="pyd" type="float"/>
    <field name="bz0x" type="float"/>
    <field name="bz0y" type="float"/>
    <field name="bz4x" type="float"/>
    <field name="bz4y" type="float"/>
    <field name="bz7x" type="float"/>
    <field name="bz7y" type="float"/>
    <field name="bz11x" type="float"/>
    <field name="bz11y" type="float"/>
    </message>'''
    f,c =np.shape(datos_static)
    t_static=np.zeros(f)
    active=np.zeros(f)
    dWP=np.zeros(f)
    pxd=np.zeros(f)
    pyd=np.zeros(f)
    t_static[:]=datos_static[:,0]
    active[:]=datos_static[:,1]
    dWP[:]=datos_static[:,2]
    pxd[:]=datos_static[:,4]
    pyd[:]=datos_static[:,5]

    return t_static,active,dWP,pxd,pyd
    
def extract_control_data(datos_control):
    '''<message name="ROVER_CTRL" id="181">
    <description> Setpoints of the rover, speed error, throttle_command, PI action and speed measured </description>
    <field name="speed_sp" type="float" unit="m/s"/>
    <!--  Speed setpoint  -->
    <field name="speed_error" type="float" unit="m/s"/>
    <!--  Speed error  -->
    <field name="throttle_command" type="float"/>
    <!--  throttle command  -->
    <field name="delta_sp" type="float" unit="deg"/>
    <!--  Angle of wheels setpoint  -->
    <field name="kp" type="float"/>
    <!--  Proporcional constant for the pid  -->
    <field name="ki" type="float"/>
    <!--  Integral constant for the pid  -->
    <field name="kd" type="float"/>
    <!--  Derivative constant for the pid  -->
    <field name="i_action" type="float"/>
    <!--  Integral action  -->
    <field name="p_action" type="float"/>
    <!--  Proportional action  -->
    <field name="d_action" type="float"/>
    <!--  Derivative action  -->
    <field name="speed_measured" type="float"/>
    <!--  Measured speed  -->
    <field name="curvature" type="float"/>
    <!--  Computed curvature with the GVF  -->
    </message>'''
    f,c =np.shape(datos_control)
    t_control=np.zeros(f)
    speed_sp=np.zeros(f)
    speed_er=np.zeros(f)
    throttle=np.zeros(f)
    delta=np.zeros(f)
    t_control[:]=datos_control[:,0]
    speed_sp[:]=datos_control[:,1]
    speed_er[:]=datos_control[:,2]
    throttle[:]=datos_control[:,4]
    delta[:]=datos_control[:,5]
     
    return t_control,speed_sp,speed_er,throttle,delta