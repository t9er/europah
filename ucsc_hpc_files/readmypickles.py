import pickle
import pdb
import matplotlib.pyplot as plt
import numpy as np


# open a file, where you stored the pickled data
file = open('C:/Users/tyler/hpc_europa/test/k2Q_omsvec_L80_N80g6_tau9_mpi_H300.pickle', 'rb')

# dump information to that file
data = pickle.load(file)

# close the file
file.close()

print('Showing the pickled data:')

cnt = 0
for item in data:
    print('The data ', cnt, ' is : ', item)
    cnt += 1

oms_size   = 1000 
oms_vec      = np.linspace(1, 3.312, oms_size)*2*np.pi/(3.5*24)/3600

plt.plot(oms_vec, data[2])
plt.show()
pdb.set_trace()