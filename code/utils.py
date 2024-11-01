import os
import pickle
from pathlib import Path


import cProfile, pstats, io
from time import time


'''
Read/write file in a pickle jar 
'''

def read_pickle_file(filename):
    try: 
        with open(filename, 'rb') as f:
            return pickle.load(f)

    except EOFError as e:
        print('Corrupted pickle ', filename)
        print(e) ## dump error 

        os.remove(filename)
        print('Deleted pickle file successfully:  ', filename)
    except:
        return None

def pickle_object(object, filename):
    directory = str(Path(filename).parent)
    os.makedirs(directory, exist_ok=True)

    with open(filename, 'wb') as f:
        pickle.dump(object, f)




def profileit(func):
    def wrapper(*args, **kwargs):
        datafn = func.__name__ + ".profile" # Name the data file sensibly
        prof = cProfile.Profile()
        retval = prof.runcall(func, *args, **kwargs)
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(prof, stream=s).sort_stats(sortby)
        ps.print_stats()
        with open(datafn, 'w') as perf_file:
            perf_file.write(s.getvalue())
        return retval

    return wrapper




def timer_func(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func