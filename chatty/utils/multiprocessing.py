import progressbar
from multiprocessing import Pool


def parmap(func, data: list, chunksize=100, processes=4, verbose=False) -> list:

    with Pool(processes) as p:
        if verbose:
            data = progressbar.progressbar(data)
        output = []
        it = p.imap(func,
                    data,
                    chunksize=chunksize)
        output = list(it)
        # for row in it:
        #     output.append(row)
        return output
