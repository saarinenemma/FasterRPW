import numpy as np
import math
from math import sqrt

def compute_euclidean_distance(coords1, coords2):
    """
    Compute the Euclidean distance between two lists of coordinates.

    Parameters
    ----------
    coords1 : ndarray
        First list of coordinates (1D array).
    coords2 : ndarray
        Second list of coordinates (1D array).

    Returns
    -------
    float
        The Euclidean distance between the two lists of coordinates.
    """
    distance = 0.0

    # Check that the input lists have the same length
    if coords1.shape[0] != coords2.shape[0]:
        raise ValueError("Both coordinate lists must have the same length")

    # Calculate the squared differences
    for i in range(coords1.shape[0]):
        diff = coords1[i] - coords2[i]
        distance += diff * diff

    # Return the square root of the sum of squared differences
    return sqrt(distance) / sqrt(coords1.shape[0])

def compute_euclidean_distances(A, B, p):
    n1 = len(A)
    n2 = len(B)
    C = np.zeros((n1, n2), dtype=np.float64)

    for i in range(len(A)):
        for j in range(len(B)):
            C[i, j] = math.pow(compute_euclidean_distance(A[i], B[j]), p)

    return C 