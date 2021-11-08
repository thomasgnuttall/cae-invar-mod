import os

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import numpy.ma as ma
import scipy.signal
import shutil

from exploration.pitch import (pitch_seq_to_cents, pitch_to_cents)
from exploration.utils import get_timestamp, myround
from exploration.io import create_if_not_exists, write_json


def plot_pitch(
    pitch, times, s_len=None, mask=None, yticks_dict=None, cents=False, 
    tonic=None, emphasize=[], figsize=None, title=None, xlabel=None, ylabel=None, 
    grid=True, ylim=None, xlim=None):
    """
    Plot graph of pitch over time

    :param pitch: Array of pitch values in Hz
    :type pitch: np.array
    :param times: Array of time values in seconds
    :type times: np.array
    :param s_len: If not None, take first <s_len> elements of <pitch> and <time> to plot
    :type s_len:  int
    :param mask: Array of bools indicating elements in <pitch> and <time> NOT to be plotted
    :type mask: np.array
    :param yticks_dict: Dict of {frequency name: frequency value (Hz)}
        ...if not None, yticks will be replaced in the relevant places with these names
    :type yticks_dict: dict(str, float)
    :param cents: Whether or not to convert frequency to cents above <tonic> 
    :type cents: bool
    :param tonic: Tonic to make cent conversion in reference to - only relevant if <cents> is True.
    :type tonic: float
    :param emphasize: list of keys in yticks_dict to emphasize on plot (horizontal red line)
    :type emphasize: list(str)
    :param figsize: Tuple of figure size values 
    :type figsize: tuple
    :param title: Title of figure, default None
    :type title: str
    :param xlabel: x axis label, default Time (s)
    :type xlabel: str
    :param ylabel: y axis label
        defaults to 'Cents Above Tonic of <tonic>Hz' if <cents>==True else 'Pitch (Hz)'
    :type ylabel: str
    :param grid: Whether to plot grid
    :type grid: bool
    :param ylim: Tuple of y limits, defaults to max and min in <pitch>
    :type ylim: bool
    :param xlim: Tuple of x limits, defaults to max and min in <time>
    :type xlim: bool

    :returns: Matplotlib objects for desired plot
    :rtype: (matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot)
    """
    if cents:
        assert tonic, \
            "Cannot convert pitch to cents without reference <tonic>"
        p1 = pitch_seq_to_cents(pitch, tonic)
    else:
        p1 = pitch

    if mask is None:
        # If no masking required, create clear mask
        mask = np.full((len(pitch),), False)
    
    if s_len:
        assert s_len <= len(pitch), \
            "Sample length is longer than length of pitch input"
    else:
        s_len = len(pitch)
        
    if figsize:
        assert isinstance(figsize, (tuple,list)), \
            "<figsize> should be a tuple of (width, height)"
        assert len(figsize) == 2, \
            "<figsize> should be a tuple of (width, height)"
    else:
        figsize = (170*s_len/186047, 10.5)

    if not xlabel:
        xlabel = 'Time (s)'
    if not ylabel:
        ylabel = f'Cents Above Tonic of {round(tonic)}Hz' \
                    if cents else 'Pitch (Hz)'

    pitch_masked = np.ma.masked_where(mask, p1)

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if grid:
        plt.grid()

    if xlim:
        xmin, xmax = xlim
    else:
        xmin = myround(min(times[:s_len]), 5)
        xmax = max(times[:s_len])
        
    if ylim:
        ymin, ymax = ylim
    else:
        sample = pitch_masked.data[:s_len]
        if not set(sample) == {None}:
            ymin_ = min([x for x in sample if x is not None])
            ymin = myround(ymin_, 50)
            ymax = max([x for x in sample if x is not None])
        else:
            ymin=0
            ymax=1000
    
    for s in emphasize:
        assert yticks_dict, \
            "Empasize is for highlighting certain ticks in <yticks_dict>"
        if s in yticks_dict:
            if cents:
                p_ = pitch_to_cents(yticks_dict[s], tonic)
            else:
                p_ = yticks_dict[s]
            ax.axhline(p_, color='#db1f1f', linestyle='--', linewidth=1)

    times_samp = times[:s_len]
    pitch_masked_samp = pitch_masked[:s_len]

    times_samp = times_samp[:min(len(times_samp), len(pitch_masked_samp))]
    pitch_masked_samp = pitch_masked_samp[:min(len(times_samp), len(pitch_masked_samp))]
    plt.plot(times_samp, pitch_masked_samp, linewidth=0.7)

    if yticks_dict:
        tick_names = list(yticks_dict.keys())
        tick_loc = [pitch_to_cents(p, tonic) if cents else p \
                    for p in yticks_dict.values()]
        ax.set_yticks(tick_loc)
        ax.set_yticklabels(tick_names)
    
    ax.set_xticks(np.arange(xmin, xmax+1, 1))

    plt.xticks(fontsize=8.5)
    ax.set_facecolor('#f2f2f2')

    ax.set_ylim((ymin, ymax))
    ax.set_xlim((xmin, xmax))

    if title:
        plt.title(title)

    return fig, ax

    
def plot_subsequence(sp, l, pitch, times, timestep, path=None, plot_kwargs={}):
    
    this_pitch = pitch[int(max(sp-l,0)):int(sp+2*l)]
    this_times = times[int(max(sp-l,0)):int(sp+2*l)]
    this_mask = this_pitch == 0
    
    fig, ax = plot_pitch(
        this_pitch, this_times, mask=this_mask,
        xlim=(min(this_times), max(this_times)), **plot_kwargs)
    
    x_d = ax.lines[-1].get_xdata()
    y_d = ax.lines[-1].get_ydata()

    x = x_d[int(min(l,sp)):int(l+min(l,sp))]
    y = y_d[int(min(l,sp)):int(l+min(l,sp))]
    
    max_y = ax.get_ylim()[1]
    min_y = ax.get_ylim()[0]
    rect = Rectangle((x_d[int(min(l,sp))], min_y), l*timestep, max_y-min_y, facecolor='lightgrey')
    ax.add_patch(rect)
    
    ax.plot(x, y, linewidth=0.7, color='darkorange')
    ax.axvline(x=x_d[int(min(l,sp))], linestyle="dashed", color='black', linewidth=0.8)

    if path:
        plt.savefig(path, dpi=90)
        plt.close('all')
    else:
        return plt



def plot_subsequence_w_stability(sp, l, pitch, time, stable_mask, timestep, path=None, plot_kwargs={}):
    this_pitch = pitch[int(max(sp-l,0)):int(sp+2*l)]
    this_times = time[int(max(sp-l,0)):int(sp+2*l)]
    this_stab = stable_mask[int(max(sp-l,0)):int(sp+2*l)]
    this_mask = this_pitch == 0

    fig, ax = plot_pitch(
        this_pitch, this_times, mask=this_mask,
        xlim=(min(this_times), max(this_times)), **plot_kwargs)

    ax2 = ax.twinx()
    ax2.plot(this_times, this_stab, 'g', linewidth=0.7, alpha=0.5, linestyle='--')

    x_d = ax.lines[-1].get_xdata()
    y_d = ax.lines[-1].get_ydata()

    x = x_d[int(min(l,sp)):int(l+min(l,sp))]
    y = y_d[int(min(l,sp)):int(l+min(l,sp))]

    max_y = ax.get_ylim()[1]
    min_y = ax.get_ylim()[0]
    rect = Rectangle((x_d[int(min(l,sp))], min_y), l*timestep, max_y-min_y, facecolor='lightgrey')
    ax.add_patch(rect)

    ax.plot(x, y, linewidth=0.7, color='darkorange')
    ax.axvline(x=x_d[int(min(l,sp))], linestyle="dashed", color='black', linewidth=0.8)
    
    if path:
        plt.savefig(path, dpi=90)
        plt.close('all')
    else:
        return plt


def plot_all_sequences(pitch, times, lengths, seq_list, direc, clear_dir=False, plot_kwargs={}):
    timestep = times[1] - times[0]
    if clear_dir:
        try:
            shutil.rmtree(direc)
        except:
            pass
    for i, seqs in enumerate(seq_list):
        for si, s in enumerate(seqs):
            l = lengths[i][si]
            l_sec = round(lengths[i][0]*timestep,1)
            t_sec = s*timestep
            str_pos = get_timestamp(t_sec)
            sp = int(s)
            plot_path = os.path.join(direc, f'motif_{i}_len={l_sec}/{si}_time={str_pos}.png')
            create_if_not_exists(plot_path)
            plot_subsequence(
                sp, l, pitch, times, timestep, path=plot_path, plot_kwargs=plot_kwargs
            )