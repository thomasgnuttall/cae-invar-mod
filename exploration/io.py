import csv
import json 
import os
import yaml

#import essentia
#import essentia.standard
import numpy as np
import librosa
import soundfile as sf

from exploration.utils import get_timestamp

def audio_loader(path, sampleRate=44100):
    """
    Load audio file from <path> to numpy array

    :param path: Path of audio file compatiuble with essentia
    :type path: str
    :param sampleRate: sample rate of audio at <path>, default 44100
    :type sampleRate: int

    :return: Array of waveform values for each timestep
    :rtype: numpy.array
    """
    loader = essentia.standard.MonoLoader(filename = path, sampleRate = sampleRate)
    audio = loader()
    return audio


def write_pitch_contour(pitch, time, path):
    """
    Write pitch contour to tsv at <path>

    :param time: Array of time values for pitch contour
    :type time: numpy.array
    :param pitch: Array of corresponding pitch values
    :type pitch: numpy.array
    :param path: path to write pitch contour to
    :type path: str
    """
    ##text=List of strings to be written to file
    with open(path,'w') as file:
        for t, p in zip(time, pitch):
            file.write(f"{t}\t{p}")
            file.write('\n')


def load_pitch_contour(path, delim=',', prat=False):
    """
    load pitch contour from tsv at <path>

    :param path: path to load pitch contour from
    :type path: str

    :return: Two numpy arrays of time and pitch values
    :rtype: tuple(numpy.array, numpy.array)
    """
    time = []
    pitch = []
    with open(path) as fd:
        rd = csv.reader(fd, delimiter=delim, quotechar='"')
        for t,p in rd:
            time.append(t)
            pitch.append(p)
    time = np.array(time).astype(float)
    pitch = np.array(pitch).astype(float)
    
    if prat:
        i1 = (pitch!=0).argmax()
        t1 = time[i1]
        d = time[i1+1] - t1

        new_times = np.flip(np.arange(-t1+d, 0, d)*-1)
        new_times = np.insert(new_times, 0, 0)
        new_pitch = np.array([0.0]*len(new_times))

        time = np.concatenate((new_times, time[i1:]))
        pitch = np.concatenate((new_pitch, pitch[i1:]))

    return time, pitch


def load_json(path):
    """
    Load json at <path> to dict
    
    :param path: path of json
    :type path: str

    :return: dict of json information
    :rtype: dict
    """ 
    # Opening JSON file 
    with open(path) as f: 
        data = json.load(f) 
    return data


def write_json(j, path):
    """
    Write json, <j>, to <path>

    :param j: json
    :type path: json
    :param path: path to write to, 
        if the directory doesn't exist, one will be created
    :type path: str
    """ 
    create_if_not_exists(path)
    # Opening JSON file 
    with open(path, 'w') as f:
        json.dump(j, f)


def load_yaml(path):
    """
    Load yaml at <path> to dictionary, d
    
    Returns
    =======
    Wrapper dictionary, D where
    D = {filename: d}
    """
    import zope.dottedname.resolve
    def constructor_dottedname(loader, node):
        value = loader.construct_scalar(node)
        return zope.dottedname.resolve.resolve(value)

    def constructor_paramlist(loader, node):
        value = loader.construct_sequence(node)
        return ParamList(value)

    yaml.add_constructor('!paramlist', constructor_paramlist)
    yaml.add_constructor('!dottedname', constructor_dottedname)

    if not os.path.isfile(path):
        return None
    with open(path) as f:
        d = yaml.load(f, Loader=yaml.FullLoader)   
    return d


def write_array(a, path):
    """
    Write array to <path>

    :param a: one dimensional array
    :type a: numpy.array
    :param path: path to write array to
    :type path: str
    """
    create_if_not_exists(path)
    with open(path, 'w') as file:
        for e in a:
            file.write(str(e))
            file.write('\n')


def read_array(path, dtype=np.float):
    """
    load array from<path>

    :param path: path to load array from
    :type path: str

    :return: array of one dimensional values
    :rtype: tuple(numpy.array, numpy.array)
    """
    arr = []
    with open(path) as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for e in rd:
            arr = arr + [dtype(e[0])]
    arr = np.array(arr, dtype=object)
    return arr


def load_tonic(path):
    """
    load tonic value frojm text file at <path>
    Text file should contain only a pitch number.

    :param path: path to load tonic from
    :type path: str

    :return: tonic in Hz
    :rtype: float
    """
    with open(path) as f:
        rd = f.read()
    return float(rd)


def create_if_not_exists(path):
    """
    If the directory at <path> does not exist, create it empty
    """
    directory = os.path.dirname(path)
    # Do not try and create directory if path is just a filename
    if (not os.path.exists(directory)) and (directory != ''):
        os.makedirs(directory)


def write_subsequences_group(y, sr, starts, lengths, timestep, output_dir):
    create_if_not_exists(output_dir)
    for i, s in enumerate(starts):
        sec_start = s*timestep
        timestamp = get_timestamp(sec_start)
        out_path = os.path.join(output_dir, f'{i}_time={timestamp}.wav')
        l = lengths[i]*sr
        s1 = sec_start*sr
        s2 = s1+l
        subseq = y[int(s1):int(s2)]
        sf.write(out_path, subseq, samplerate=sr)


def write_all_sequence_audio(audio_path, all_seqs, all_lens, timestep, output_dir):
    y, sr = librosa.load(audio_path)
    for i, seq in enumerate(all_seqs):
        out_dir = os.path.join(output_dir, f'motif_{i}/')
        lens = [l*timestep for l in all_lens[i]]
        write_subsequences_group(y, sr, seq, lens, timestep, out_dir)


def load_if_exists(path, dtype=float):
    if os.path.exists(path):
        a = np.loadtxt(path, dtype=dtype)
    else:
        a = None
    return a

