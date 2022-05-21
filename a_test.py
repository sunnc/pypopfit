import numpy as np
import time

nn=10000000
aa=np.arange(nn)

t1=time.time()
s=0
for ii in range(nn):
	s=s+aa[ii]
t2=time.time()
print(t2-t1)

t1=time.time()
s=0
for ii in range(len(aa)):
	s=s+aa[ii]
t2=time.time()
print(t2-t1)
