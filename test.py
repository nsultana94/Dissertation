import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

array = np.array([1,2,3,4,5,6])
arr = sliding_window_view(array, window_shape = 3)
print(len(arr))