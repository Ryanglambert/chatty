import progressbar
from multiprocessing import Pool


def parmap(func, data: list, chunksize=100, processes=4, verbose=False) -> list:

    with Pool(processes) as p:
        if verbose:
            return p.map(func,
                         progressbar.progressbar(data),
                         chunksize=chunksize)
        else:
            return p.map(func, data, chunksize=chunksize)
