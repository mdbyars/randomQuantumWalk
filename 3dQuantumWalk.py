# -*- coding: utf-8 -*-
# @author: Thomas Anderson
 
###################
#   quantum.py    #
###################
# Thomas Anderson #
###################
#    00/00/0000   #
###################
#  Last updated:  #
#    21/03/2014   #
###################

#################################
#  Quantum Random Walk Project  #
#################################

# Import numpy and matplotlib.pyplot
import numpy as np, matplotlib.pyplot as plt, matplotlib as mpt
from mpl_toolkits.mplot3d import Axes3D # Import 3D plotting
import matplotlib.animation as animatio
#################
# Begin Program #
#################

############################################################
# Some of the following is adapted from Susan Stepney from #
# http://susan-stepney.blogspot.co.uk/2014/02/mathjax.html #
############################################################

# Define the positions and number of steps
N = 20      # number of random steps
P = 2*N+1    # number of positions

# Define the two states a quantum coin can be in a superposition of
coin0 = np.array([1, 0])  # |0>
coin1 = np.array([0, 1])  # |1>

# Calculate all the outer products required for the Hademard operator
C00 = np.outer(coin0, coin0)  # |0><0| 
C01 = np.outer(coin0, coin1)  # |0><1| 
C10 = np.outer(coin1, coin0)  # |1><0| 
C11 = np.outer(coin1, coin1)  # |1><1| 

# Define the Hademart coin operator
C_hat = (C00 + C01 + C10 - C11)/np.sqrt(2.)

# Define the step operator and the two possible shifts
ShiftPlus = np.roll(np.eye(P), 1, axis=0)
ShiftMinus = np.roll(np.eye(P), -1, axis=0)
S_hat = np.kron(ShiftPlus, C00) + np.kron(ShiftMinus, C11)

# From all of these, define the walk operator
U = S_hat.dot(np.kron(np.eye(P), C_hat)) # This is the operation to be repeated

# Set the time
time = np.arange(N+1)

# Set the values of all positions on the line to 0 except the initial position
posn0 = np.zeros(P)
posn0[N] = 1 # 2N+1 Positions, index starts at 0 so middle is at N

psi0 = np.kron(posn0,(coin0+coin1*1j)/np.sqrt(2.)) # Set the initial coin state

prob = np.empty((N+1,P)) # Initialise the array of probabilities

for j in time:
    
    psiN = np.linalg.matrix_power(U, j).dot(psi0) # Work out the state of the system after N walks
    
    # Iterate through each position
    for k in range(P):
        posn = np.zeros(P) # Set values of all positions to zero
        posn[k] = 1 # Except current position
        M_hat_k = np.kron( np.outer(posn,posn), np.eye(2)) # Perform a measurement at the current position
        proj = M_hat_k.dot(psiN) # Find out what state the system is it at the position
        prob[j][k] = proj.dot(proj.conjugate()).real # Calculate the probability of the particle being here
      
# Set the x and y values
xval, yval = np.arange(P), time

# Mesh the x and y values together
xval, yval = np.meshgrid(xval, yval)

# Begin plotting the graph
fig = plt.figure() # Create an overall figure
ax = fig.add_subplot(111, projection='3d') # Add a 3D plot

# NOTE: Only plots non-zero values
ax.plot_surface(xval, yval, prob, rstride=1, cstride=1, vmin=0, vmax=0.1, cmap=mpt.cm.coolwarm) # Plot the data
loc = range (int(0), int(P), int(P / 10)) #Location of ticks
plt.xticks(loc) # Set the x axis ticks
plt.xlim(0, P) # Set the limits of the x axis
ax.set_xticklabels(range (int(-N), int(N+1), int(P / 10))) # Set the labels of the x axis
plt.xlabel("Position") # Set x label
plt.ylabel("Time") # Set y label
ax.set_zlabel("Probability") # Set z label

plt.show() # Show the graph

nfr = 30 # Number of frames
fps = 10 # Frame per sec
xs = []
ys = []
zs = []
ss = np.arange(1,nfr,0.5)
for s in ss:
    xs.append(normal(50,s,200))
    ys.append(normal(50,s,200))
    zs.append(normal(50,s,200))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sct, = ax.plot([], [], [], "o", markersize=2)
def update(ifrm, xa, ya, za):
    sct.set_data(xa[ifrm], ya[ifrm])
    sct.set_3d_properties(za[ifrm])
ax.set_xlim(0,100)
ax.set_ylim(0,100)
ax.set_zlim(0,100)
ani = animation.FuncAnimation(fig, update, nfr, fargs=(xs,ys,zs), interval=1000/fps)
fn = 'plot_3d_scatter_funcanimation'
ani.save(fn+'.mp4',writer='ffmpeg',fps=fps)
ani.save(fn+'.gif',writer='imagemagick',fps=fps)
import subprocess
cmd = 'magick convert %s.gif -fuzz 5%% -layers Optimize %s_r.gif'%(fn,fn)
subprocess.check_output(cmd)
###############
# End Program #
###############
