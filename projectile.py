import math
import numpy as np
from scipy.integrate import odeint
from functools import partial

def projectile_kernel(vartuple, t, g=1.0, gamma=0.0):
    x, vx, y, vy, z, vz = vartuple
    derivatives = [vx, -gamma*vx, vy, -gamma*vy, vz, -g-gamma*vz]
    return derivatives

def find_landing_pos(sol, t):
    x, vx, y, vy, z, vz = np.transpose(sol)
    tidx = len(filter(lambda h: h>0, z))
    landed_t = t[tidx] - z[tidx]*(t[tidx+1]-t[tidx])/(z[tidx+1]-z[tidx])
    landed_x = x[tidx] - z[tidx]*(x[tidx+1]-x[tidx])/(z[tidx+1]-z[tidx])
    landed_y = y[tidx] - z[tidx] * (y[tidx + 1] - y[tidx]) / (z[tidx + 1] - z[tidx])
    return landed_x, landed_y, landed_t, tidx

def solve_projectile(vx0, vy0, vz0, x0=0.0, y0=0.0, z0=0.0, g=1.0, gamma=0.0, nbsteps=1000):
    maxT = (vz0 + math.sqrt(vz0*vz0+2*g*y0))/g
    t = np.linspace(0, maxT, nbsteps)

    var0 = [x0, vx0, y0, vy0, z0, vz0]
    sol = odeint(partial(projectile_kernel, g=g, gamma=gamma), var0, t)

    landed_x, landed_y, landed_t, tidx = find_landing_pos(sol, t)

    results = {'landed_x': landed_x,
               'landed_y': landed_y,
               'landed_t': landed_t,
               'sol_array': sol,
               't_array': t,
               'landed_tidx': tidx}

    return results