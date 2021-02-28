# Author: Styfen Schaer <schaers@student.ethz.ch> 

import cmath
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import time


@njit(parallel=True)
def julia(c, maxit, maxabs, xrange, yrange, xpixel, ypixel):
    """ Computes the julia set. We use numba to accelarate the computation.
    Remove the njit decorate and change "prange" to "range" to compute without numba. """
    image = np.zeros((xpixel, ypixel))
    xstep = (xrange[1] - xrange[0]) / xpixel
    ystep = (yrange[1] - yrange[0]) / ypixel
    for x in prange(xpixel):
        for y in prange(ypixel):
            z0 = complex(xrange[0] + x*xstep, yrange[0] + y*ystep)
            z = z0**2 + c
            it = 0
            while abs(z) <= maxabs and it <= maxit:
                z = z**2 + c
                it += 1
            image[x, y] = it
    return image


if __name__ == '__main__':
    """ (Initial) conditions and settings.
    - c: starting value
    - MAXIT: maximal number of interations
    - MAXABS: criterion for divergence
    - xrange: x-range where the jualias-set is computet
    - yrange: y-range where the jualias-set is computet
    - XPIXEL: number of point computet in x-direction
    - YPIXEL: number of point computet in y-direction """
    c = complex(-0.1, -0.65)
    MAXIT = 1024
    MAXABS = 10
    xrange = np.array([-1.5, 1.5])
    yrange = xrange
    XPIXEL = 10000
    YPIXEL = XPIXEL

    """ We first compute a small julia-set to jit compile the code (that's what numba does). 
    So we can see the full speed with the next function call next. """
    julia(c=complex(-0.194, -0.6657), maxit=256, maxabs=10, xrange=np.array([-2, 2]), yrange=np.array([-2, 2]), xpixel=10, ypixel=10)

    """ Enjoy how fast numba computes the julia-set """
    tic = time.time()
    julia_matrix = julia(c=c, maxit=MAXIT, maxabs=MAXABS, xrange=xrange, yrange=yrange, xpixel=XPIXEL, ypixel=YPIXEL)
    print('This took: {0} seconds'. format(round(time.time() - tic, 3)))

    """ Plot and save the image (~40x40 cm in this case).
    You may want to change the pixels, colormap and dpi. """
    fig, ax = plt.subplots(figsize=(16, 16))
    
    ax.imshow(julia_matrix, cmap='cividis')
    
    ax.axis('off')
    plt.savefig('images/julia-set.jpeg', dpi=300, quality=95, optimize=True))

