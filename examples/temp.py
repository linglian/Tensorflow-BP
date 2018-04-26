import Queue
import numpy as np
import multiprocessing
import time

def tempFun(msg):
    time.sleep(1)
    print msg

testFl = [0, 1, 2, 3, 4, 5, 6]

t_time = time.time()
pool = multiprocessing.Pool(processes=4)
result = []
result.append(pool.apply_async(tempFun, (2131231, )))
result.append(pool.apply_async(tempFun, (2131232, )))
result.append(pool.apply_async(tempFun, (2131233, )))
result.append(pool.apply_async(tempFun, (2131234, )))

pool.close()
pool.join()

for i in result:
    print i.get()
print 'Speed Time: %d ' % (time.time() - t_time)

