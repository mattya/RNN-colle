from numpy import *
import struct

NTRAIN=1000000
NTEST=10000

def write_binary(a, filename):
    f = open(filename, "wb")
    if len(a.shape)==1:
        for j in range(a.shape[0]):
            f.write(struct.pack("i", int(a[j])))
    else:
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                f.write(struct.pack("i", int(a[i,j])))
    f.close()


"""
in0 010010101011001100100010110100101011010110
in1 000001000000000000000000000000000000000000
in2 000000000000001000000000000000000000000000
in3 000000000000000000000000000010000000000000
         #########
out 000000000000000000000000000000101011000000
"""

def recall_task(tmax, lenmax):
    x_in = zeros((tmax, 4))
    x_out = zeros(tmax)

    t_start = random.randint(0, tmax/4)
#    t_end = random.randint(t_start+lenmax/2, t_start+lenmax)
    t_end = t_start+5
    ln = t_end-t_start
    t_recall = random.randint(t_end+1, tmax-ln-1)
#    t_recall = t_end+50

    for i in range(tmax):
        x_in[i,0] = random.randint(0, 2)

    x_in[t_start,1] = 1
    x_in[t_end,2] = 1
    x_in[t_recall,3] = 1

    x_out[t_recall+1:t_recall+ln+1] = x_in[t_start:t_end,0]

    return x_in, x_out, t_start, t_end, t_recall

NB = 50
NP = 10

x_train_in_stacked = zeros((NTRAIN, 4))
x_train_out_stacked = zeros(NTRAIN)
x_train_heads = []
for i in range(NTRAIN/NB):
    x_in, x_out, t_start, t_end, t_recall = recall_task(NB, NP)
    x_train_in_stacked[i*NB:(i+1)*NB,:] = x_in
    x_train_out_stacked[i*NB:(i+1)*NB] = x_out
    x_train_heads.append(i*NB)

x_test_in_stacked = zeros((NTEST, 4))
x_test_out_stacked = zeros(NTEST)
x_test_heads = []
for i in range(NTEST/NB):
    x_in, x_out, t_start, t_end, t_recall = recall_task(NB, NP)
    x_test_in_stacked[i*NB:(i+1)*NB,:] = x_in
    x_test_out_stacked[i*NB:(i+1)*NB] = x_out
    x_test_heads.append(i*NB)
#    print x_in[i,0],x_in[i,1],x_in[i,2],x_in[i,3],x_out[i]
#print x_in_stacked
write_binary(x_train_in_stacked, "../data/recall/recall_train_in.txt")
write_binary(x_train_out_stacked, "../data/recall/recall_train_out.txt")
write_binary(x_test_in_stacked, "../data/recall/recall_test_in.txt")
write_binary(x_test_out_stacked, "../data/recall/recall_test_out.txt")