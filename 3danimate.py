import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
 
# References
# https://gist.github.com/neale/e32b1f16a43bfdc0608f45a504df5a84
# https://towardsdatascience.com/animations-with-matplotlib-d96375c5442c
# https://riptutorial.com/matplotlib/example/23558/basic-animation-with-funcanimation
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

##########################################################
#plt.style.use('seaborn-pastel')

###############
# End Program #
###############

# ANIMATION FUNCTION
def func(num, dataSet, line):
    # NOTE: there is no .set_data() for 3 dim data...
    line.set_data(dataSet[0:2, :num])    
    line.set_3d_properties(dataSet[2, :num])    
    return line
 
 
# THE DATA POINTS
t = time # This would be the z-axis ('t' means time here)
x = xval
y = yval
dataSet = np.array([x, y, t])
numDataPoints = len(t)
 
# GET SOME MATPLOTLIB OBJECTS
fig = plt.figure()
ax = Axes3D(fig)
 
# NOTE: Can't pass empty arrays into 3d version of plot()
line = plt.plot(t, x, y, lw=2, c='g')[0] # For line plot
 
# AXES PROPERTIES]
# ax.set_xlim3d([limit0, limit1])
ax.set_xlabel('X(t)')
ax.set_ylabel('Y(t)')
ax.set_zlabel('time')
ax.set_title('Trajectory of electron for E vector along [120]')
 
# Creating the Animation object
line_ani = animation.FuncAnimation(fig, func, frames=numDataPoints, fargs=(dataSet,line), interval=50, blit=False)
#line_ani.save(r'AnimationNew.mp4')
 
 
plt.show()