from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np  # Add this import
from matplotlib import style
style.use('fivethirtyeight')
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

# Convert your lists to numpy arrays and create a meshgrid
X = np.array([1,2,3,4,5,6,7,8,9,10])
Y = np.array([2,3,5,5,6,7,4,4,7,7])
Z = np.array([[1,2,4,6,8,5,3,2,5,7]])

#X, Y = np.meshgrid(x, y)

# Create a corresponding Z matrix
# This is just an example - adjust the calculation based on what you want to visualize
              #for _ in range(len(y))])
X2 = np.array([1,-2,3,-4,-5,6,-7,8,-9,10])
Y2 = np.array([-2,-3,-5,-5,-6,-7,-4,-4,-7,-7])
Z2 = np.array([1,2,4,6,8,5,3,2,5,7])

X3 = np.array([1,2,3,4,5,6,7,8,9,10])
Y3 = np.array([2,3,5,5,6,7,4,4,7,7])
Z3 = np.array([0,0,0,0,0,0,0,0,0,0])
dx = np.ones(10)
dy = np.ones(10)
dz = np.array([1,2,3,4,5,6,7,8,9,10])

# Plot the wireframe
#ax1.plot_wireframe(X, Y, Z)
# ax1.scatter(X,Y,Z, c='g', marker='o')
# ax1.scatter(X2,Y2,Z2, c='r', marker='o')
ax1.bar3d(X3, Y3, Z3, dx, dy, dz)

ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
ax1.set_zlabel('z axis')  # Uncommented this line

plt.show()