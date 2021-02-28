# Author: Styfen Schaer <schaers@student.ethz.ch> 

import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numba import njit


@njit
def barnsley_fern(n_points):
    """ Computes the baernsley fern. We use numba to accelarate the computation.
    Comment out the njit decorate to compute it without numba. """
    f1 = lambda z: complex(0 * z.real + 0 * z.imag + 0, 0 * z.real + 0.16 * z.imag + 0)
    f2 = lambda z: complex(0.85 * z.real + 0.04 * z.imag + 0, -0.04 * z.real + 0.85 * z.imag + 1.60)
    f3 = lambda z: complex(0.2 * z.real + -0.26 * z.imag + 0, 0.23 * z.real + 0.22 * z.imag + 1.60)
    f4 = lambda z: complex(-0.15 * z.real + 0.28 * z.imag + 0, 0.26 * z.real + 0.24 * z.imag + 1.44)

    z = 0
    rand_nums = np.random.rand(n_points)
    zx, zy = np.empty(n_points), np.empty(n_points)
    count = 0
    for n in rand_nums:
        if n <= 0.01:
            z = f1(z)
            zx[count] = z.real
            zy[count] = z.imag
        elif 0.86 >= n > 0.01:
            z = f2(z)
            zx[count] = z.real
            zy[count] = z.imag
        elif 0.93 >= n > 0.86:
            z = f3(z)
            zx[count] = z.real
            zy[count] = z.imag
        else:
            z = f4(z)
            zx[count] = z.real
            zy[count] = z.imag
        count += 1
    return zx, zy


if __name__ == '__main__':
    """ Number of points to compute """
    NPOINTS = 10000000

    """ We first compute a small barnsley fern to jit compile the code (that's what numba does). 
    So we can see the full speed with the next function call next. """
    barnsley_fern(n_points=10)

    """ Enjoy how fast numba computes the barnsley fern """
    tic = time.time()
    zx, zy = barnsley_fern(n_points=NPOINTS)
    print('This took: {0} seconds'. format(round(time.time() - tic, 3)))

    """ Plot and save the image.
    You may want to change the marker size of the plot and the dpi and size of the figure to just play around. """
    fig, ax = plt.subplots(figsize=(16, 16))

    cmap = plt.get_cmap('viridis', 1024)
    norm = mpl.colors.Normalize(vmin=0, vmax=10)
    colors = cmap(norm(zy))  # we have a color gradient along the y-axis
    ax.scatter(zx, zy, c=colors, marker='.', s=0.01, lw=0)

    fig.patch.set_facecolor('k')
    ax.axis('off')
    plt.savefig('images/barnsley-fern.jpeg', dpi=600, quality=95, optimize=True)
