import dtaidistance
from experiments.alapana_dataset_analysis.dtw import dtw_path
from matplotlib.ticker import NullFormatter, FormatStrFormatter
import matplotlib.pyplot as plt

i = 36
j = 34

sr=44100

r=0.1

def plot_dtw(pat1, pat2, path_dtw, dtw_norm, r, write):
	plt.close()

	create_if_not_exists(write)
	nullfmt = NullFormatter()

	# definitions for the axes
	left, width = 0.12, 0.60
	bottom, height = 0.08, 0.60
	bottom_h =  0.16 + width 
	left_h = left + 0.27 
	rect_plot = [left_h, bottom, width, height]
	rect_x = [left_h, bottom_h, width, 0.2]
	rect_y = [left, bottom, 0.2, height]

	# start with a rectangular Figure
	plt.figure(2, figsize=(8, 8))

	axplot = plt.axes(rect_plot)
	axx = plt.axes(rect_x)
	axx.grid()
	axy = plt.axes(rect_y)
	axy.grid()
	# Plot the matrix
	#axplot.pcolor(acc.T,cmap=cm.gray)
	axplot.plot([x[0] for x in path_dtw], [x[1] for x in path_dtw], 'black')

	axplot.set_xlim((0, len(pat1)))
	axplot.set_ylim((0, len(pat2)))
	axplot.tick_params(axis='both', which='major', labelsize=18)

	# Plot time serie horizontal
	axx.plot(pat1,'.', color='k')
	axx.tick_params(axis='both', which='major', labelsize=18)
	xloc = plt.MaxNLocator(4)
	x2Formatter = FormatStrFormatter('%d')
	axx.yaxis.set_major_locator(xloc)
	axx.yaxis.set_major_formatter(x2Formatter)

	# Plot time serie vertical
	axy.plot(pat2, range(len(pat2)),'.',color='k')
	axy.invert_xaxis()
	yloc = plt.MaxNLocator(4)
	xFormatter = FormatStrFormatter('%d')
	axy.xaxis.set_major_locator(yloc)
	axy.xaxis.set_major_formatter(xFormatter)
	axy.tick_params(axis='both', which='major', labelsize=18)

	# Limits
	axx.set_xlim(axplot.get_xlim())
	axy.set_ylim(axplot.get_ylim())

	plt.title(f'r={r}, dtw={round(dtw_norm,2)}')
	plt.savefig(write)
	plt.close()


def get_derivative(pitch, time):

    d_pitch = np.diff(pitch) / np.diff(time)
    d_time = (np.array(time)[:-1] + np.array(time)[1:]) / 2

    return d_pitch, d_time

all_groups[all_groups['index']==i].iloc[0]['start']

qstart = all_groups[all_groups['index']==i].iloc[0]['start']
qend = all_groups[all_groups['index']==i].iloc[0]['end']
qtrack = all_groups[all_groups['index']==i].iloc[0]['track']
qindex = i

(qpitch, qtime, qpitchstep) = pitch_tracks[qtrack]
diff_qpitch, diff_qtime = get_derivative(qpitch, qtime)

all_groups[all_groups['index']==j].iloc[0]['start']

rstart = all_groups[all_groups['index']==j].iloc[0]['start']
rend = all_groups[all_groups['index']==j].iloc[0]['end']
rtrack = all_groups[all_groups['index']==j].iloc[0]['track']
rindex = j

(rpitch, rtime, rpitchstep) = pitch_tracks[rtrack]
diff_rpitch, diff_rtime = get_derivative(rpitch, rtime)

pat1 = qpitch[int(qstart/qtimestep):int(qend/qtimestep)]
pat2 = rpitch[int(rstart/rtimestep):int(rend/rtimestep)]

pat1_time = qtime[int(qstart/qtimestep):int(qend/qtimestep)]
pat2_time = rtime[int(rstart/rtimestep):int(rend/rtimestep)]

#diff_pat1 = diff_qpitch[int(qstart/qtimestep):int(qend/qtimestep)]
#diff_pat2 = diff_rpitch[int(rstart/rtimestep):int(rend/rtimestep)]

diff_pat1,_ = get_derivative(pat1,pat1_time)
diff_pat2,_ = get_derivative(pat2,pat2_time)


p1l = len(pat1)
p2l = len(pat2)

l_longest = max([p1l, p2l])
l_shortest = min([p1l, p2l])

#if l_longest/l_shortest-1 > 0.5:
#    continue

path, dtw_val = dtw_path(pat1, pat2, radius=round(l_longest*r))
#dtw_val = dtaidistance.dtw.distance(pat1, pat2, window=round(l_longest*r), use_c=True, psi=round(l_longest*r))

l = len(path)
dtw = dtw_val/l


plot_dtw(pat1, pat2, path, dtw, r, f'plots/dtw/DIFF_DTW_i={i}__j={j}.png')

p1l = len(diff_pat1)
p2l = len(diff_pat2)

l_longest = max([p1l, p2l])
l_shortest = min([p1l, p2l])
#if l_longest/l_shortest-1 > 0.5:
#    continue
path, dtw_val = dtw_path(diff_pat1, diff_pat2, radius=round(l_longest*r))
#dtw_val = dtaidistance.dtw.distance(diff_pat1, diff_pat2, window=round(l_longest*r), use_c=True, psi=round(l_longest*r))

l = len(path)
diff_dtw = dtw_val/l

plot_dtw(diff_pat1, diff_pat2, path, diff_dtw, r, f'plots/dtw/DTW_i={i}__j={j}.png')



is1 = [x[0] for x in path]
is2 = [x[1] for x in path]

join_path = list(zip(diff_pat1[is1], diff_pat2[is2]))

accum = [(x,y,(abs(x-y)**2)**0.5) for x,y in join_path]


path = path[:10]
pat1 = pat1[:10]
pat2 = pat2[:10]

sum([abs(pat1[x[0]]-pat2[x[1]])**2 for x in path])**0.5/len(path)
 

distances = np.abs(pat1[path[:, 0]] - pat2[path[:, 1]]) ** 2
np.sqrt(np.sum(distances)) / len(path)