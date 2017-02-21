import numpy as np

x1 = np.ones((3,4))
x2 = np.arange(12)
x2 = x2.reshape((3,4))

print("x1",x1,"\nx2",x2,"\nx1.T",x1.T,"x2.T\n",x2.T)

print(np.matmul(x1,x2.T))
