import matplotlib.pyplot as plt

import random

right_indices = all_groups[all_groups['handedness']=='Right']['index'].values
left_indices = all_groups[all_groups['handedness']=='Left']['index'].values

fig = plt.figure()
ax = plt.axes(projection ='3d')

i=random.choice(right_indices)
j=random.choice(left_indices)

feature='3dpositionDTWHand'

i_vec = index_features[i][feature]
j_vec = index_features[j][feature]

xi = i_vec[:,0]
yi = i_vec[:,1]
zi = i_vec[:,2]

xj = j_vec[:,0]
yj = j_vec[:,1]
zj = j_vec[:,2]

# plotting
ax.plot3D(xi, yi, zi, c='blue', label=f'index={i}')
 
# plotting
ax.plot3D(xj, yj, zj, c='red', label=f'index={j}')

ax.plot(0,0,0,'ro')
plt.legend()
ax.set_ylim((-0.4,0.4))
ax.set_xlim((-0.4,0.4))
ax.set_zlim((-0.4,0.4))
ax.set_ylabel('y')
ax.set_xlabel('x')
ax.set_zlabel('z')
plt.savefig('3d_plot.png')
plt.close('all')