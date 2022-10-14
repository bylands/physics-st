import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

st.header('Kepler Ellipse')

#%% Define differential equations
def func(t, y):
    r = np.sqrt(y[0]*y[0]+y[1]*y[1])
    f = -GM/r**3
    return [y[2], y[3], f*y[0], f*y[1]]

#%% Define parameters
AE = 149.5978707e9

planets_data = {
    'Mercury': [[69.82e9, 0, 0, 38.9e3], 88./365],
    'Earth': [[152.1e9, 0, 0, 29290.61952], 1.],
    'Pluto': [[49.304*AE, 0, 0, 3.74e3], 260.],
    'Halley': [[35.14*AE, 0, 0, 0.91152215e3], 75.]
}

list_of_planets = set(planets_data.keys())

planets = st.multiselect('Select a planet', list_of_planets, 'Earth')

GM = 1.32712442099e20 # gravitationel parameter Sun

#%% Solve differential equation
tmax = max([planets_data[p][1] for p in planets if p in planets_data])
step = 0.01*10**(max(np.ceil(np.log10(tmax)), 0))

years = st.slider('Simulation time (Earth years)', step, 10*tmax, tmax, step=step)

t_final = years*365*86400 # simulation time
t_span = [0, t_final]
t_eval = np.linspace(0, t_final, int(years*1000))

sol = {}

for planet in planets:

    sol[planet] = solve_ivp(func, t_span, planets_data[planet][0], t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-6)

#%% Plot solution

smin = min([planets_data[p][0][0] for p in planets if p in planets_data])/AE
smax = max([planets_data[p][0][0] for p in planets if p in planets_data])/AE
fmin = 10**(np.floor(np.log10(smin)))
fmax = 10**(np.floor(np.log10(smax)))
rmin = np.floor(smin/fmin)*fmin
rmax = 2*np.ceil(smax/fmax)*fmax
rval = float(np.ceil(10.5*smax/fmax)*fmax/10)

size = st.slider('Plot range in AE', rmin, rmax, rval, fmin/10)
fig, ax = plt.subplots()

ax.scatter(0, 0, marker='.', s=50, c="red") # draw Sun

for planet in planets:
    ax.plot(sol[planet].y[0]/AE, sol[planet].y[1]/AE, c="blue") # plot ellipse

plt.xlim(-size, size)
plt.ylim(-size, size)

ax.set_aspect(1) # set aspect ratio to 1

col1, col2, col3 = st.columns([1, 4, 1])

with col2:
    st.pyplot(fig=fig)
