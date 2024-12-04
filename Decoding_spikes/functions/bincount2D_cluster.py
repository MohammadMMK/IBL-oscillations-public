import numpy as np
import pandas as pd

def bincount2D_cluster(x, y, xbin, ybin, xlim, yscale):
    """
    Modified version of iIBL bincount2D for binnning cluster. this  version ensures output size matches the provided `yscale`.

    :param x: values to bin along the 2nd dimension (time).
    :param y: values to bin along the 1st dimension (clusters).
    :param xbin: scalar, bin size along x.
    :param ybin: ignored in this implementation (bins defined by yscale).
    :param xlim: 2 values (array or list) restricting the range along x.
    :param yscale: all unique cluster IDs to enforce consistent binning.
    :return: 3 numpy arrays MAP [ny,nx] image, xscale [nx], yscale [ny].
    """
    # Define the x scale
    xscale = np.arange(xlim[0], xlim[1] + xbin / 2, xbin)

    # Map y (clusters) to indices of yscale
    y_indices = np.searchsorted(yscale, y)

    # Initialize the output array
    nx, ny = len(xscale), len(yscale)
    counts = np.zeros((ny, nx), dtype=np.float32)

    # Bin the data
    x_indices = np.floor((x - xlim[0]) / xbin).astype(np.int64)
    valid = (x_indices >= 0) & (x_indices < nx) & (y_indices >= 0) & (y_indices < ny)
    np.add.at(counts, (y_indices[valid], x_indices[valid]), 1)

    return counts, xscale, yscale