from numpy import *

NTRAIN=1000000
NTEST=10000

def narma(tmax, k):
    v = random.randn(tmax)
    x = zeros(tmax)

    for i in range(1,tmax):
        for k in range(max(0,i-k), i):
            x[i] += 0.004*x[i-1]*x[k]
        x[i] += 0.2*x[i-1] + 0.01
        if i>=k:
            x[i] += 1.5*v[i]*v[i-k]

    return v, x

v, x = narma(NTRAIN, 30)
savetxt("../data/narma/narma_train_in.txt", v, fmt="%.4f")
savetxt("../data/narma/narma_train_out.txt", x, fmt="%.4f")
v, x = narma(NTEST, 30)
savetxt("../data/narma/narma_test_in.txt", v, fmt="%.4f")
savetxt("../data/narma/narma_test_out.txt", x, fmt="%.4f")
