import numpy as np

# Implements https://doi.org/10.1007/s004540010063


def edgewise_simplex_subdivision(x, k=1):
  n = len(x)
  d = x.shape[1]

  cumsum = np.cumsum(x * k, axis=1)

  # get the boundaries and keep their associated colors
  boundaries = np.where(cumsum[:, :-1] != k, cumsum[:, :-1] % 1, 1)
  # last color should have cumsum = 1
  boundaries = np.concatenate([boundaries, np.ones((n, 1))], axis=1)
  boundaries_c = np.tile(np.arange(d), (n, 1))

  # sort the boundaries
  I = np.argsort(boundaries, axis=1, stable=True)
  boundaries = np.take_along_axis(boundaries, I, axis=1)
  boundaries_c = np.take_along_axis(boundaries_c, I, axis=1)

  # cutoffs at which a color is yield
  cutoffs = boundaries[:, None, :] + np.arange(k)[None, :, None]
  cutoffs = cutoffs.reshape(n, -1)
  # in case some rows do not sum to 1
  cutoffs = np.concatenate([cutoffs[:, :-1], cumsum[:, -1][:, None]], axis=1)

  # count the number of occurrences of each color
  counts = [np.zeros(n, dtype=int)]

  for i in range(d):

    # cumulative count for colors 1 to i
    c = (cutoffs < cumsum[:, i][:, None]).sum(axis=1)

    # handle ties
    ties = cutoffs == cumsum[:, i][:, None]
    ind = np.where(boundaries_c == i)[1]
    mask = np.arange(d)[None, :] <= ind[:, None]
    c += (ties & np.tile(mask, (1, k))).sum(axis=1)

    counts.append(c - counts.pop(-1))
    counts.append(c)  # also remember the cumulative count

  counts = counts[:-1]

  # C[j, i] is the number of times color i appears in M[j], the color scheme
  # of the j-th example
  C = np.stack(counts, axis=1)
  return C


# Taken from https://gist.github.com/mblondel/c99e575a5207c76a99d714e8c6e08e89


def projection_simplex(V, z=1, axis=None):
  """
  Projection of x onto the simplex, scaled by z:
      P(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2
  z: float or array
      If array, len(z) must be compatible with V
  axis: None or int
      axis=None: project V by P(V.ravel(); z)
      axis=1: project each V[i] by P(V[i]; z[i])
      axis=0: project each V[:, j] by P(V[:, j]; z[j])
  """
  if axis == 1:
    n_features = V.shape[1]
    U = np.sort(V, axis=1)[:, ::-1]
    z = np.ones(len(V)) * z
    cssv = np.cumsum(U, axis=1) - z[:, np.newaxis]
    ind = np.arange(n_features) + 1
    cond = U - cssv / ind > 0
    rho = np.count_nonzero(cond, axis=1)
    theta = cssv[np.arange(len(V)), rho - 1] / rho
    return np.maximum(V - theta[:, np.newaxis], 0)

  elif axis == 0:
    return projection_simplex(V.T, z, axis=1).T

  else:
    V = V.ravel().reshape(1, -1)
    return projection_simplex(V, z, axis=1).ravel()
