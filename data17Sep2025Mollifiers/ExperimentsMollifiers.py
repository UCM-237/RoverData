#      Copyright (C) 2025  Alfredo Gonzalez Calvin
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
plt.rcParams['figure.dpi'] = 120
plt.rcParams.update({
        'font.size': 18,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

# We need to normalize so that \int \psi(x) = 1
def psi(x):
    if(np.size(x) == 1):
        if(np.abs(x) < 1):
            return np.exp(-1/(1-x**2)) / 0.44399
        return 0
    
    karg =  np.argwhere(np.abs(x) < 1)
    y = np.zeros(np.size(x))
    y[karg] = np.exp(-1/(1-x[karg]**2)) * 1 / 0.44399
    return y
    #if(np.abs(x) < 1):
    #    if(np.abs(1 - x**2) < np.finfo(float).eps):
    #        return 0.0
    #    else:
    #        return 1 / 0.44399 * np.exp(- 1 / (1-x**2))
    #return 0.0
    
def psi_eps(x,epsilon):
    return psi(x/epsilon) / epsilon

# In each dimension..
def func_lines(points, gamma, n_seg):
    
    integer_part = int(gamma)
    fractional_part = gamma - integer_part
    if(gamma <= 1):
        return (1-fractional_part) * points[0] + fractional_part * points[1]
    elif(integer_part < n_seg):
        return (1 - fractional_part) * points[integer_part] + fractional_part * points[integer_part+1]
    else:
        return ( (1 - (gamma - n_seg + 1)) * points[n_seg - 1] +
                (gamma - n_seg + 1) * points[n_seg]);
    
    
def convolutionx(x, points, epsilon, n_seg):
    
    points_of_integration = 100
    lower_integration_value = -epsilon
    upper_integration_value = epsilon
    step_of_integration = (upper_integration_value - lower_integration_value) / points_of_integration
    
    convolution_at_point = 0
    step = 0
    for k in range(points_of_integration):
        step = k  * step_of_integration
        
        convolution_at_point = convolution_at_point + (
            func_lines(points, x-(lower_integration_value+step), n_seg) * 
            psi_eps(lower_integration_value + step, epsilon))
    
    return convolution_at_point * step_of_integration


plt.close('all')

data_rover = pd.read_table(r"25_09_17__11_27_58_5.csv")

# Constants in order to convert ins data
int32_frac_speed = 19;
int32_frac_accel = 10;
int32_frac_pos = 8;

# Choose time instants
t_first  = [178, 281]
t_second = [309, 400]
t_third  = [417, 533]
#t_fourth = [1214, 1305.7]

## Fetch data

# Time (s)
time = np.array(data_rover['Time'])

# Position in x (m)
pos_x = np.array(data_rover['INS:ins_x'])/(1 << int32_frac_pos)

# Position in y (m)
pos_y = np.array(data_rover['INS:ins_y'])/(1 << int32_frac_pos)

# GVF w parameter
w = np.array(data_rover['GVF_PARAMETRIC:w'])

# GNSS accuracy in position (m)
gps_acc_pos = np.array(data_rover['GPS_INT:pacc']) / 100

# Fetch data relative to the points
data_points = np.array(data_rover['GVF_PARAMETRIC:p'])

# Errors
phi_errors  = np.array(data_rover['GVF_PARAMETRIC:phi'])

phi_errors_real = np.zeros((len(phi_errors),2))   # Two dimensional error
data_points_real = np.zeros((len(phi_errors),16)) # Telemetry buffer

for k in range(len(phi_errors)):
    phi_errors_real[k,:]  = np.fromstring(phi_errors[k],count=2, sep=',') # Make it numpy array
    data_points_real[k,:] = np.fromstring(data_points[k], count=16, sep=',')

# Get X points and Y points of the trajectory
index_y_points = np.argwhere(data_points_real[:,0]<0)[-1][-1]
index_x_points = np.argwhere(data_points_real[:,0]>0)[-1][-1]
num_segments = int(data_points_real[index_x_points][0])

points_trajectory_x = data_points_real[index_x_points][1:num_segments+2]
points_trajectory_y = data_points_real[index_y_points][1:num_segments+2]

# Create the mollified trajectory knowing the epsilon values
epsilons = np.array([0.5, 1.0, 0.2])
num_points_plot = 200
molli_domain_values = np.linspace(0, num_segments, num_points_plot)

# Num_points x Dimensions x Number of epsilons
mollifiers_values = np.zeros((num_points_plot, 2, len(epsilons)))
for k in range(len(epsilons)):
    for j in range(num_points_plot):
        mollifiers_values[j,0,k] = convolutionx(molli_domain_values[j], points_trajectory_x, epsilons[k], num_segments)
        mollifiers_values[j,1,k] = convolutionx(molli_domain_values[j], points_trajectory_y, epsilons[k], num_segments)

original_fun = np.zeros((num_points_plot,2))
for k in range(num_points_plot):
    original_fun[k,0] = func_lines(points_trajectory_x, molli_domain_values[k], num_segments)
    original_fun[k,1] = func_lines(points_trajectory_y, molli_domain_values[k], num_segments)

# From telemetry
L = 0.04

## Show data of first trajectory
t_index_first = (time >= t_first[0]) & (time < t_first[1])

plt.figure(figsize=(12,8))
plt.subplot(121)
plt.plot(points_trajectory_y, points_trajectory_x, 'kx', markersize=10)
plt.plot(original_fun[:,1], original_fun[:,0], 'k', label=r"$f$")
plt.plot(mollifiers_values[:,1,0], mollifiers_values[:,0,0], 'r', label=r"$F$")
plt.plot(pos_y[t_index_first], pos_x[t_index_first], 'b', label=r"$r(t;\pi(\xi_0))$")
plt.xlabel(r"$(m)$")
plt.ylabel(r"$(m)$")
plt.legend()
plt.title("Trajectory and position of the rover \n $\\varepsilon_1=\\varepsilon_2=0.5$")
plt.axis('equal')

plt.subplot(122)
plt.plot(time[t_index_first] - time[t_index_first][0], phi_errors_real[t_index_first,0] / L, 'r', label=r"$F_1(\Phi(t,\xi_0)_3) - r_1(t;\pi(\xi_0))$")
plt.plot(time[t_index_first] - time[t_index_first][0], phi_errors_real[t_index_first,1] / L, 'b', label=r"$F_2(\Phi(t,\xi_0)_3)-r_2(t;\pi(\xi_0))$")
#plt.plot(time[t_index_first], np.abs(pos_y[t_index_first] -mollifiers_values[] ))
plt.ylim([-3,2])
max_error = 3
#plt.plot([time[t_index_first][0], time[t_index_first][-1]] - time[t_index_first][0], max_error *np.ones(2), 'k--')
#plt.plot([time[t_index_first][0], time[t_index_first][-1]] - time[t_index_first][0], -max_error *np.ones(2), 'k--')
plt.plot(time[t_index_first] - time[t_index_first][0], gps_acc_pos[t_index_first], 'g', label=r"GNSS acc")
plt.plot(time[t_index_first] - time[t_index_first][0], -gps_acc_pos[t_index_first], 'g')
plt.xlabel(r"Time $(s)$")
plt.ylabel(r"Errors $(m)$")
plt.title("GNSS and trajectory errors \n $\\varepsilon_1=\\varepsilon_2=0.5$")
plt.legend()
plt.tight_layout()


## Show data of second trajectory
t_index_second = (time >= t_second[0]) & (time < t_second[1])

plt.figure(figsize=(12,8))
plt.subplot(121)
plt.plot(points_trajectory_y, points_trajectory_x, 'kx', markersize=10)
plt.plot(original_fun[:,1], original_fun[:,0], 'k', label=r"$f$")
plt.plot(mollifiers_values[:,1,1], mollifiers_values[:,0,1], 'r', label=r"$F$")
plt.plot(pos_y[t_index_second], pos_x[t_index_second], 'b', label=r"$r(t;\pi(\xi_0))$")
plt.xlabel(r"$(m)$")
plt.ylabel(r"$(m)$")
plt.legend()
plt.title("Trajectory and position of the rover \n $\\varepsilon_1=\\varepsilon_2=1.0$")
plt.axis('equal')

plt.subplot(122)
plt.plot(time[t_index_second] - time[t_index_second][0], phi_errors_real[t_index_second,0] / L, 'r', label=r"$F_1(\Phi(t,\xi_0)_3) - r_1(t;\pi(\xi_0))$")
plt.plot(time[t_index_second] - time[t_index_second][0], phi_errors_real[t_index_second,1] / L, 'b', label=r"$F_2(\Phi(t,\xi_0)_3)-r_2(t;\pi(\xi_0))$")
#plt.plot(time[t_index_first], np.abs(pos_y[t_index_first] -mollifiers_values[] ))
plt.ylim([-3,2])
max_error = 3
#plt.plot([time[t_index_second][0], time[t_index_second][-1]] - time[t_index_second][0], max_error *np.ones(2), 'k--')
#plt.plot([time[t_index_second][0], time[t_index_second][-1]] - time[t_index_second][0], -max_error *np.ones(2), 'k--')
plt.plot(time[t_index_second] - time[t_index_second][0], gps_acc_pos[t_index_second], 'g', label=r"GNSS acc")
plt.plot(time[t_index_second] - time[t_index_second][0], -gps_acc_pos[t_index_second], 'g')
plt.xlabel(r"Time $(s)$")
plt.ylabel(r"Errors $(m)$")
plt.title("GNSS and trajectory errors \n $\\varepsilon_1=\\varepsilon_2=1.0$")
plt.legend()
plt.tight_layout()

## Show data of third trajectory
t_index_third = (time >= t_third[0]) & (time < t_third[1])

plt.figure(figsize=(12,8))
plt.subplot(121)
plt.plot(points_trajectory_y, points_trajectory_x, 'kx', markersize=10)
plt.plot(original_fun[:,1], original_fun[:,0], 'k', label=r"$f$")
plt.plot(mollifiers_values[:,1,2], mollifiers_values[:,0,2], 'r', label=r"$F$")
plt.plot(pos_y[t_index_third], pos_x[t_index_third], 'b', label=r"$r(t;\pi(\xi_0))$")
plt.xlabel(r"$(m)$")
plt.ylabel(r"$(m)$")
plt.legend()
plt.title("Trajectory and position of the rover \n $\\varepsilon_1=\\varepsilon_2=0.2$")
plt.axis('equal')

plt.subplot(122)
plt.plot(time[t_index_third] - time[t_index_third][0], phi_errors_real[t_index_third,0] / L, 'r', label=r"$F_1(\Phi(t,\xi_0)_3) - r_1(t;\pi(\xi_0))$")
plt.plot(time[t_index_third] - time[t_index_third][0], phi_errors_real[t_index_third,1] / L, 'b', label=r"$F_2(\Phi(t,\xi_0)_3)-r_2(t;\pi(\xi_0))$")
#plt.plot(time[t_index_first], np.abs(pos_y[t_index_first] -mollifiers_values[] ))
plt.ylim([-3,5])
max_error = 3
#plt.plot([time[t_index_third][0], time[t_index_third][-1]] - time[t_index_third][0], max_error *np.ones(2), 'k--')
#plt.plot([time[t_index_third][0], time[t_index_third][-1]] - time[t_index_third][0], -max_error *np.ones(2), 'k--')
plt.plot(time[t_index_third] - time[t_index_third][0], gps_acc_pos[t_index_third], 'g', label=r"GNSS acc")
plt.plot(time[t_index_third] - time[t_index_third][0], -gps_acc_pos[t_index_third], 'g')

plt.xlabel(r"Time $(s)$")
plt.ylabel(r"Errors $(m)$")
plt.title("GNSS and trajectory errors \n $\\varepsilon_1=\\varepsilon_2=0.2$")
plt.legend()
plt.tight_layout()
