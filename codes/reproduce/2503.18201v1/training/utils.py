import numpy as np

def set_seed(seed):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
