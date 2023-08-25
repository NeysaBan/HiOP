import torch
import time
import numpy as np

def show_time(func, cuda = True, device = "cuda:1", ntest = 10):
    print()
    times = list()
    res = None
    # GPU warm up
    for _ in range(ntest):
        func()
    for _ in range(ntest):
        # sync the threads to get accurate cuda running time
        torch.cuda.synchronize(device)
        start_time = time.time()
        res = func()
        torch.cuda.synchronize(device)
        end_time = time.time()
        times.append((end_time-start_time)*1e6)

    print("time cost:  {:.3f}us".format(np.mean(times)))
    
    return res


