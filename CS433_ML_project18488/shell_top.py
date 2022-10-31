import os
from threading import Thread
import threading
import time


lambda_=[0,0.1,0.5,1,5,10]
deg=[4,5,7,10,11,12]
gamma=[0.01,0.05,0.2,0.5]

def exe_proc(i,j,k):
    command='python run.py {} {} {} {} {} {} {} {} {}'.format(lambda_[i],deg[j],gamma[k],lambda_[i],deg[j],gamma[k],lambda_[i],deg[j],gamma[k],)
    os.system(command)
model_nums=3

#writing_lock=threading.Lock()
for i in range(len(lambda_)):
    
    for j in range(len(deg)):
        for k in range(len(gamma)):
            Threads_grid=[]
            Threads_grid.append(Thread(target=exe_proc, args=(i,j,k)))
            for thread_iter in Threads_grid:
                thread_iter.start()
            for thread_iter in Threads_grid:
                thread_iter.join()