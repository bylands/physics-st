import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

st.header('Kepler Areas')

#%% Define differential equations
def func(t, y):
    r = np.sqrt(y[0]*y[0]+y[1]*y[1])
    f = -GM/r**3
    return [y[2], y[3], f*y[0], f*y[1]]

#%% Define parameters
AE = 149.5978707e9
year = 365*86400

planets_data = {
    'Mercury': [[69.82e9, 0, 0, 38.9e3], 88./365],
    'Earth': [[152.1e9, 0, 0, 29290.61952], 1.],
    'Pluto': [[49.304*AE, 0, 0, 3.74e3], 260.],
    'Halley': [[35.14*AE, 0, 0, 0.91152215e3], 75.]
}

list_of_planets = list(planets_data.keys())
N_list = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
powers_of_ten = {10:'10', 100:'100', 1000:'1\'000', 10000:'10\'000', \
    100000:'100\'000', 1000000:'1\'000\'000', 10000000:'10\'000\'000'}


col1, col2, col3 = st.columns(3)

with col1:
    planet = st.selectbox('Select a planet', list_of_planets, 1)

with col2:
    N = st.selectbox('# of steps in simulation', N_list, 2, \
    format_func=lambda x: powers_of_ten[x])

angle_points_list = []
for i in range(1, int(np.log10(N))+1):
    angle_points_list.append(int(10**i))
angle_points_list.reverse()

with col3:
    angle_numbers = st.selectbox('Number of angles', angle_points_list, \
        format_func=lambda x: powers_of_ten[x])
    angle_steps = int(N/angle_numbers)

GM = 1.32712442099e20 # gravitationel parameter Sun

#%% Solve differential equation
years = planets_data[planet][1]

t_final = years*year # simulation time
t_span = [0, t_final]
t_eval = np.linspace(0, t_final, N+1)

col1, col2, col3 = st.columns(3)

with col1:
    rtol = st.selectbox('relative tolerance', [1e-4, 1e-6, 1e-8, 1e-10], 1)

with col2:   
    atol = st.selectbox('absolute tolerance', [1e-4, 1e-6, 1e-8, 1e-10], 1)

sol = solve_ivp(func, t_span, planets_data[planet][0], t_eval=t_eval, \
    method='RK45', rtol=rtol, atol=atol)

#%% Calculate areas
a = (max(sol.y[0])-min(sol.y[0]))/2
b = (max(sol.y[1])-min(sol.y[1]))/2

t_eval = t_eval[np.ceil(angle_steps/2)::angle_steps]

x0 = sol.y[0]
y0 = sol.y[1]

x1 = np.roll(x0, -1)
y1 = np.roll(y0, -1)

th = np.arctan2(y0, x0)

r0 = np.sqrt(x0*x0+y0*y0)
r1 = np.sqrt(x1*x1+y1*y1)

da = (x0*y1 - x1*y0)/2

area = [da[i:i+angle_steps].sum() for i in range(0, da.size, angle_steps)]

av_sim = np.average(area)
av_th = a*b*np.pi/angle_numbers
dev = (av_sim-av_th)/av_th

# st.write(f'average area: {av_sim:.5e}, theoretical average: {av_th:.5e}, deviation: {dev:.2e}')

#%% Plot areas

area_rc = area[:-1]/area[0]-1 # relative change of area

fig, ax = plt.subplots()

ax.scatter(t_eval/year, area_rc, marker=".", s=5, c='green')

ymax = 10**np.ceil(np.log10(max(max(area_rc), -min(area_rc))))

plt.ylim(-ymax, ymax)

col1, col2, col3 = st.columns([1, 4, 1])

with col2:
    st.pyplot(fig=fig)
