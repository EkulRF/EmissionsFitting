import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import pickle as pkl
import scipy

from radis import calc_spectrum
from radis import load_spec
from radis.test.utils import getTestFile
from radis import Spectrum

from Toolbox_Processing import *
from Toolbox_Reading import *
from Toolbox_Inversion import *

x_sol = np.load('sol.npy')
sigma = np.load('sig.npy')
Compounds = np.load('comp.npy', allow_pickle=True).item()
ref_spec = np.load('ref.npy')
obs_spec = np.load('obs.npy')

print(Compounds.keys())



with open('C.pickle', 'rb') as handle:
    C = pkl.load(handle)
#plt.imshow(C)
#plt.savefig('C.jpg')

cobj = spl.spilu(C)
# c = cobj.to_dense()
# c = np.linalg.inv(cobj.L.toarray())
#plt.imshow(c, interpolation="nearest", cmap=plt.get_cmap("RdBu"))
#plt.savefig('C.jpg')

# #plt.show()
# plt.figure()

# plt.spy(c)
# plt.savefig('C_test.jpg')

#cobj = spl.spilu(C)



Ns, Nl = ref_spec.shape
Nt = obs_spec.shape[0]

s = np.zeros(Ns * Nt)
t = np.zeros(Ns * Nt)
for i in range(Ns * Nt):
    a = t * 1.0
    a[i] = 1.0
    s[i] = cobj.solve(a)[i]

plt.spy(s)
#plt.imshow(s, interpolation="nearest", cmap=plt.get_cmap("RdBu"))
plt.savefig('C.jpg')
