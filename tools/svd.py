import sys
from numpy import *
m = sys.argv[1]
n = sys.argv[2]
tauinv = float(sys.argv[3])
fn = sys.argv[4]


#print(n)
#print(m)
#print(fn)
a = random.randn(int(n),int(m))
if int(n)==int(m):
    a = tauinv*a + (1.0-tauinv)*identity(int(n))
s,v,d = linalg.svd(a)
v = zeros(a.shape)
for i in range(min(int(n), int(m))):
    v[i,i] = 1.0

#print(s.shape)
#print(v.shape)
#print(d.shape)
b = dot(s,dot(v,d))

if int(n)>int(m):
    b *= sqrt(float(n)/float(m))

savetxt(fn, b.T, fmt="%.4e")
