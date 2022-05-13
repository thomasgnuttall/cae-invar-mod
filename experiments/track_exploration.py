import sys
sys.path.append('../')
import os
import numpy as np
import skimage.io
def prepro(X):
    X = X - np.median(X)
    return X

def create_if_not_exists(path):
    """
    If the directory at <path> does not exist, create it empty
    """
    directory = os.path.dirname(path)
    # Do not try and create directory if path is just a filename
    if (not os.path.exists(directory)) and (directory != ''):
        os.makedirs(directory)

def load_sim_matrix(path):
    X = np.load(path)
    X = prepro(X)
    return X

def main(track_name):
    ## Get Data
    out_dir = os.path.join(f'/Volumes/Shruti/asplab2/cae-invar/data/self_similarity/{track_name}/')

    sim_path = os.path.join(out_dir, 'self_sim.npy')

    print(f'Loading self similarity from {sim_path}')
    X = load_sim_matrix(sim_path)

    length = X.shape[0]

    bounds = list(np.arange(0, length, 4000)) + [length-1]

    break_points = [(bounds[i],bounds[i+1]) for i in range(len(bounds[1:]))]

    ####################
    ## Load sim array ##
    ####################
    for s1, s2 in break_points:
        X_samp = X.copy()[s1:s2,s1:s2]

        sim_filename = os.path.join('/Volumes/Shruti/asplab2/track_exploration/', f'{track_name}_{s1}_{s2}.png')

        create_if_not_exists(sim_filename)
        
        skimage.io.imsave(sim_filename, X_samp)

if __name__ == '__main__':
    track_name = sys.argv[1]
    main(track_name)