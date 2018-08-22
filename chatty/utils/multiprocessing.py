from multiprocessing import Pool


def parmap(func, data: list, chunksize=100, processes=4) -> list:
    with Pool(processes) as p:
        return p.map(func, data, chunksize=chunksize)