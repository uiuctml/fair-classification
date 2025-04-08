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
