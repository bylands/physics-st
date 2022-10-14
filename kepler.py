import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
from datetime import timedelta


# Definition Differentialgleichungen
def func(t, y):
    r = np.sqrt(y[0]*y[0]+y[1]*y[1])
    f = -GM/r**3
    return [y[2], y[3], f*y[0], f*y[1]]

# y0 = [152.1e9, 0, 0, 29290.61952] # Anfangsbedingungen Erde
y0 = [69.82e9, 0, 0, 38.9e3] # Anfangsbedingungen Merkur

GM = 1.32712442099e20 # Gravitationsparameter der Sonne

years = 10
t_final = years*365*86400 # simulation time
t_span = [0, t_final]
t_eval = np.linspace(0, t_final, int(years*100))

start = time.monotonic()

sol = solve_ivp(func, t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-6)

end = time.monotonic()
exectime = timedelta(seconds=end-start).total_seconds()
print("\nexecution time: {} s".format(exectime))

# plot solution
fig, ax = plt.subplots()
ax.plot(sol.y[0], sol.y[1], c="blue")
ax.set_aspect(1)
# ax.axis('equal')

ax.scatter(0, 0, marker='.', s=50, c="red")

plt.show()