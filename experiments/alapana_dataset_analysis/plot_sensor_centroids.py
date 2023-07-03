import matplotlib.pyplot as plt

import random

fig = plt.figure()
ax = plt.axes(projection ='3d')



i = 30

pelvis=all_groups[all_groups['index']==i]['Pelvis_centroid'].values[0]

angle=all_groups[all_groups['index']==i]['angle'].values[0]

rightHand=all_groups[all_groups['index']==i]['RightHand_centroid'].values[0]
rightHand = (rightHand[0]-pelvis[0], rightHand[1]-pelvis[1], rightHand[2]-pelvis[2])

leftHand=all_groups[all_groups['index']==i]['LeftHand_centroid'].values[0]
leftHand = (leftHand[0]-pelvis[0], leftHand[1]-pelvis[1], leftHand[2]-pelvis[2])

head=all_groups[all_groups['index']==i]['Head_centroid'].values[0]
head = (head[0]-pelvis[0], head[1]-pelvis[1], head[2]-pelvis[2])

rightShoulder=all_groups[all_groups['index']==i]['RightShoulder_centroid'].values[0]
rightShoulder = (rightShoulder[0]-pelvis[0], rightShoulder[1]-pelvis[1], rightShoulder[2]-pelvis[2])

leftShoulder=all_groups[all_groups['index']==i]['LeftShoulder_centroid'].values[0]
leftShoulder = (leftShoulder[0]-pelvis[0], leftShoulder[1]-pelvis[1], leftShoulder[2]-pelvis[2])

rightHand = rotate(rightHand, -angle)
leftHand = rotate(leftHand, -angle)
head = rotate(head, -angle)
rightShoulder = rotate(rightShoulder, -angle)
leftShoulder = rotate(leftShoulder, -angle)

pelvis = (0,0,0)

ax.plot(pelvis[0],pelvis[1],pelvis[2],'ro', label='pelvis')
ax.plot(rightHand[0],rightHand[1],rightHand[2],'bo', label='rightHand')
ax.plot(leftHand[0],leftHand[1],leftHand[2],'go', label='leftHand')
ax.plot(head[0],head[1],head[2],'co', label='head')
ax.plot(rightShoulder[0],rightShoulder[1],rightShoulder[2],'mo', label='rightShoulder')
ax.plot(leftShoulder[0],leftShoulder[1],leftShoulder[2],'yo', label='leftShoulder')
plt.legend()
#ax.set_ylim((-10,10))
#ax.set_xlim((-10,10))
#ax.set_zlim((-10,10))
ax.set_ylabel('y')
ax.set_xlabel('x')
ax.set_zlabel('z')
plt.savefig('3d_plot.png')
plt.close('all')