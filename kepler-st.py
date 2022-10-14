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

col1, col2 = st.columns([1,2])

with col1:
    option = st.selectbox('Select a planet', ('Mercury', 'Earth', 'Pluto', 'Halley'), index=1)

    match option:
        case 'Mercury':
            y0 = [69.82e9, 0, 0, 38.9e3] # initial conditions Mercury
            t0 = 88./365
        case 'Earth':
            y0 = [152.1e9, 0, 0, 29290.61952] # initial conditions Earth
            t0 = 1.
        case 'Pluto':
            y0 = [49.304*AE, 0, 0, 3.74e3] # initial conditions Earth
            t0 = 260.
        case 'Halley':
            y0 = [35.14*AE, 0, 0, 0.91152215e3] # initial conditions Earth
            t0 = 75.
            

with col2:
    st.write('')
    st.write('')
    st.write(f'initial position (aphel): {y0[0]/AE:.3f} AE,\
        aphel velocity: {y0[3]/1e3:.3f} km/s')

step = 0.01*10**(max(np.ceil(np.log10(t0)), 0))

years = st.slider('Simulation time (Earth years)', step, 10*t0, t0, step=step)

GM = 1.32712442099e20 # gravitationel parameter Sun

t_final = years*365*86400 # simulation time
t_span = [0, t_final]
t_eval = np.linspace(0, t_final, int(years*1000))

#%% Solve differential equation
sol = solve_ivp(func, t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-6)

#%% Plot solution
fig, ax = plt.subplots()

ax.scatter(0, 0, marker='.', s=50, c="red") # draw Sun

ax.plot(sol.y[0]/AE, sol.y[1]/AE, c="blue") # plot ellipse

ax.set_aspect(1) # set aspect ratio to 1
max = y0[0]/AE*1.1
plt.xlim([-max, max])
plt.ylim([-max, max])

col1, col2, col3 = st.columns([1, 4, 1])

with col2:
    st.pyplot(fig=fig)
