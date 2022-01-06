
print('Cleaning isolated non-directional regions using morphological opening')
X_binop = apply_bin_op(X_fill, binop_dim)

# clean up


#   pool = Pool(os.cpu_count())

#   # Add your data to the datasplit variable below:
#   indices = [(i,j) for i,j in itertools.combinations(range(h),r=2) if i <= j]

#   results = pool.map(lambda i,j: get_centered_array(X_cont, i, j, filter_size), indices)

#   pool.close()
#   pool.join()

#   from multiprocessing import Pool
#   import os
#   import numpy as np
#   import tqdm

#   def get_centered_array(X, x, y, s):
#       """
#       Return <s>x<s> array centered on <x> and <y> in <X>
#       Any part of returned array that exists outside of <X>
#       is be filled with nans
#       """
#       o, r = np.divmod(s, 2)
#       l = (x-(o+r-1)).clip(0)
#       u = (y-(o+r-1)).clip(0)
#       X_ = X[l: x+o+1, u:y+o+1]
#       out = np.full((s, s), np.nan, dtype=X.dtype)
#       out[:X_.shape[0], :X_.shape[1]] = X_
#       return out


#   def is_surrounded(X):
#       """
#       Is the center square in x sufficiently surrounded by 
#       non zero 
#       """
#       triu = np.triu(X)
#       tril = np.tril(X)
#       np.fill_diagonal(triu, 0)
#       np.fill_diagonal(tril, 0)
#       return 1 in triu and 1 in tril


#   filter_size = 3

#   h, w = X_cont.shape

#   X_fill = np.zeros((h, w))

#   for i in tqdm.tqdm(range(h)):
#       for j in range(w):
#           if i < j:
#               continue            
#           cent_X = get_centered_array(X_cont, i, j, filter_size)
#           if is_surrounded(cent_X):
#               X_fill[i,j] = 1


#   X_fill = X_fill + X_fill.T - np.diag(np.diag(X_fill))

#   if save_imgs:
#       skimage.io.imsave(merg_filename, X_fill)


print('Applying Hough Transform')
peaks = hough_transform_new(X_fill, hough_high_angle, hough_low_angle, hough_threshold, filename=hough_filename)

#sprint('Averaging very close Hough lines')
#X_fill, averaged_peaks  = group_and_fill_hough(X_cont, peaks, merg_filename)

#shough_filename = os.path.join(out_dir, '7_Koti Janmani_hough.png') if save_imgs else None
#shough_av_filename = os.path.join(out_dir, '7b_Koti Janmani_hough_av.png') if save_imgs else None

#splot_hough_new(X_cont, peaks, hough_filename)
#splot_hough_new(X_cont, averaged_peaks, hough_av_filename)

