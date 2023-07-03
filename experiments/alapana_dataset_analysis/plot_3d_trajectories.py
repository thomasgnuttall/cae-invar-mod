import matplotlib.pyplot as plt

import random

right_indices = all_groups[all_groups['handedness']=='Right']['index'].values
left_indices = all_groups[all_groups['handedness']=='Left']['index'].values

fig = plt.figure()
ax = plt.axes(projection ='3d')

i=random.choice(right_indices)
level='Position'
feature='Hand'
hand=all_groups[all_groups['index']==i]['handedness'].values[0]
num_dims=3
pelvis=all_groups[all_groups['index']==i]['Pelvis_centroid'].values[0]
rightHand=all_groups[all_groups['index']==i]['RightHand_centroid'].values[0]
leftHand=all_groups[all_groups['index']==i]['LeftHand_centroid'].values[0]
head=all_groups[all_groups['index']==i]['Head_centroid'].values[0]
rightShoulder=all_groups[all_groups['index']==i]['RightShoulder_centroid'].values[0]
leftShoulder=all_groups[all_groups['index']==i]['LeftShoulder_centroid'].values[0]

assert num_dims in [1,2,3], '<num_dims> must be 1,2,3'
assert level in ['Position', 'Velocity', 'Acceleration'], '<level> must be either Position, Velocity or Acceleration'
feature_name = hand+feature
d = idict(i)
track_name = d['track']
end = d['end']
start = d['start']
mp = get_mocap_path(track_name, substring=level)
df = mocap[mp]
min_t = df['time_ms'].iloc[0]

if level == 'Position' and min_t==0:
    df = df[1:]
    min_t = df['time_ms'].iloc[0]

df['time_s'] = df['time_ms'].apply(lambda y: (y-min_t)/1000)

this_frame = df[(df['time_s']>=start) & (df['time_s']<=end)]
this_frame = this_frame.loc[:, (slice(None), feature_name)]

# defining coordinates for the 2 points.
x_orig = np.copy(this_frame['x'][feature_name].values)
y_orig = np.copy(this_frame['y'][feature_name].values)
z_orig = np.copy(this_frame['z'][feature_name].values)
 
# plotting
#ax.plot3D(x_orig, y_orig, z_orig, c='green', label='original')

if level == 'Position':
    this_frame['x'] = this_frame['x'].apply(lambda y: y-pelvis[0])
    this_frame['y'] = this_frame['y'].apply(lambda y: y-pelvis[1])
    this_frame['z'] = this_frame['z'].apply(lambda y: y-pelvis[2])

# defining coordinates for the 2 points.
x_pelvis = np.copy(this_frame['x'][feature_name].values)
y_pelvis = np.copy(this_frame['y'][feature_name].values)
z_pelvis = np.copy(this_frame['z'][feature_name].values)
 
# plotting
#ax.plot3D(x_pelvis, y_pelvis, z_pelvis, c='red', label='pelvis_norm')

if hand =='Left':
    this_frame.loc[:,('x',feature_name)] = -this_frame.loc[:,('x',feature_name)]

# defining coordinates for the 2 points.
x_mirror = np.copy(this_frame['x'][feature_name].values)
y_mirror = np.copy(this_frame['y'][feature_name].values)
z_mirror = np.copy(this_frame['z'][feature_name].values)
 
# plotting
ax.plot3D(x_mirror, y_mirror, z_mirror, c='blue', label=f'index={i} [Right]')





i=random.choice(left_indices)
level='Position'
feature='Hand'
hand=all_groups[all_groups['index']==i]['handedness'].values[0]
num_dims=3
pelvis=all_groups[all_groups['index']==i]['Pelvis_centroid'].values[0]
rightHand=all_groups[all_groups['index']==i]['RightHand_centroid'].values[0]
leftHand=all_groups[all_groups['index']==i]['LeftHand_centroid'].values[0]
head=all_groups[all_groups['index']==i]['Head_centroid'].values[0]
rightShoulder=all_groups[all_groups['index']==i]['RightShoulder_centroid'].values[0]
leftShoulder=all_groups[all_groups['index']==i]['LeftShoulder_centroid'].values[0]

assert num_dims in [1,2,3], '<num_dims> must be 1,2,3'
assert level in ['Position', 'Velocity', 'Acceleration'], '<level> must be either Position, Velocity or Acceleration'
feature_name = hand+feature
d = idict(i)
track_name = d['track']
end = d['end']
start = d['start']
mp = get_mocap_path(track_name, substring=level)
df = mocap[mp]
min_t = df['time_ms'].iloc[0]

if level == 'Position' and min_t==0:
    df = df[1:]
    min_t = df['time_ms'].iloc[0]

df['time_s'] = df['time_ms'].apply(lambda y: (y-min_t)/1000)

this_frame = df[(df['time_s']>=start) & (df['time_s']<=end)]
this_frame = this_frame.loc[:, (slice(None), feature_name)]

# defining coordinates for the 2 points.
x_orig = np.copy(this_frame['x'][feature_name].values)
y_orig = np.copy(this_frame['y'][feature_name].values)
z_orig = np.copy(this_frame['z'][feature_name].values)
 
# plotting
#ax.plot3D(x_orig, y_orig, z_orig, c='green', label='original')

if level == 'Position':
    this_frame['x'] = this_frame['x'].apply(lambda y: y-pelvis[0])
    this_frame['y'] = this_frame['y'].apply(lambda y: y-pelvis[1])
    this_frame['z'] = this_frame['z'].apply(lambda y: y-pelvis[2])

# defining coordinates for the 2 points.
x_pelvis = np.copy(this_frame['x'][feature_name].values)
y_pelvis = np.copy(this_frame['y'][feature_name].values)
z_pelvis = np.copy(this_frame['z'][feature_name].values)
 
# plotting
#ax.plot3D(x_pelvis, y_pelvis, z_pelvis, c='red', label='pelvis_norm')

if hand =='Left':
    this_frame.loc[:,('x',feature_name)] = -this_frame.loc[:,('x',feature_name)]

# defining coordinates for the 2 points.
x_mirror = np.copy(this_frame['x'][feature_name].values)
y_mirror = np.copy(this_frame['y'][feature_name].values)
z_mirror = np.copy(this_frame['z'][feature_name].values)
 
# plotting
ax.plot3D(x_mirror, y_mirror, z_mirror, c='red', label=f'index={i} [Left]')




ax.plot(0,0,0,'ro')
plt.legend()
#ax.set_ylim((-10,10))
#ax.set_xlim((-10,10))
#ax.set_zlim((-10,10))
ax.set_ylabel('y')
ax.set_xlabel('x')
ax.set_zlabel('z')
plt.savefig('3d_plot.png')
plt.close('all')