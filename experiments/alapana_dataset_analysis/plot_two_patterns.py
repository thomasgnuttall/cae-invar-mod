def plot_index(i, path='test_plot.png'):
    track = all_patts[all_patts['index']==i].iloc[0]
    track_name = track.track
    start = track.start
    end = track.end
    tonic = track.tonic

    pitch, time, timestep =  pitch_tracks[track_name]

    start_seq = int(start/timestep)
    end_seq = int(end/timestep)
    length_seq = end_seq - start_seq

    plot_kwargs = {
        'figsize':(15*2/3,4*2/3),
        'ylim': (0,3000)
    }

    plot_subsequence(start_seq, length_seq, pitch, time, timestep, path=path, plot_kwargs=plot_kwargs)


i=9
j=10
def plot_both(i, j):
    plot_index(i)
    plot_index(j, path='test_plot2.png')




    