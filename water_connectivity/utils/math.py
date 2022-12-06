from scipy import ndimage


def laplace_periodic(u):
    """
    Laplace operator on a 2D matrix with periodic boundary conditions
    """
    return ndimage.laplace(u, mode='wrap')
