import vpython as vp
import time
import random
import numpy as np
from krill import *

def rastrigin(x, y):
    return 20 + x**2 - 10*np.cos(2*np.pi*x) + y**2 - 10*np.cos(2*np.pi*y)

def schwefel(x,y):
    return 418.9829*2 - x*np.sin(np.sqrt(np.abs(x))) - y*np.sin(np.sqrt(np.abs(y)))

def ackley(x,y):
    return -20 * np.exp(-0.2*np.sqrt((1/2)*(x**2+y**2))) - np.exp((1/2)*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y))) + 20 + np.exp(1)

def peak(x,y):
    return x*np.exp(-(x**2+y**2))

def exponential(x,y):
    return -np.exp(-(x**2+y**2))

def rosenbrock(x,y):
    return 100*(y-x**2)**2 + (x-1)**2

scene = vp.canvas(title='Testando KH', width=1920, height=880, center=vp.vector(0,0,0))

limits = 1
func = 'exponential'

if func == 'schwefel':
    limits = 500
    f=schwefel
elif func == 'rastrigin':
    limits = 5.12
    f=rastrigin
elif func == 'ackley':
    limits = 32.768
    f=ackley
elif func == 'peak':
    limits = 2
    f=peak
elif func == 'exponential':
    limits = 5
    f=exponential
elif func == 'rosenbrock':
    limits = 2.048
    f=rosenbrock
    
xmin, ymin = -limits, -limits
xmax, ymax = limits, limits
pts = 60
dt = (xmax-xmin)/pts

vertexes = []
z_max = -np.inf
z_min = np.inf
x_points = np.arange(xmin, xmax, dt)
y_points = np.arange(ymin, ymax, dt)
for i in x_points:
    for j in y_points:
        z = f(i,j)
        if z>z_max:
            z_max = z
        elif z<z_min:
            z_min = z

z_abs = max(np.abs(z_min), np.abs(z_max))
vertexes = []
for i in x_points:
    vertex_line = []
    for j in y_points:
        z = f(i,j)
        z_normalize = 5*z/z_abs
        x_normalize = 5*(-1+2*(i-xmin)/(xmax-xmin))
        y_normalize = 5*(-1+2*(j-ymin)/(ymax-ymin))
        pos_v = vp.vector(x_normalize, y_normalize, z_normalize)
        
        red = z-z_min/(z_max-z_min)
        blue = 1 - red
        green = 0.1
        vertex_line.append(vp.vertex(pos = pos_v, color = vp.vector(red,green,blue)))
    vertexes.append(vertex_line)
    
surfaces = [] 
for i in range(0,pts-1):
    for j in range(0,pts-1):
        surfaces.append(vp.quad(vs = [vertexes[i][j], vertexes[i+1][j], vertexes[i+1][j+1], vertexes[i][j+1]]))
        

n_iteracoes=1000
swarm = Swarm(30, limits, z_abs, f, n_iteracoes)
x=0
while x<n_iteracoes:
    vp.rate(5)
    swarm.evaluate_fitness()
    swarm.food_position()
    for i in range(swarm.size):
        swarm.change_position(i, x)