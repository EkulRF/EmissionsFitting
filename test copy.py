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
C = np.load('C.npy', allow_pickle=True)
cobj = spl.spilu(C.any())
c = np.linalg.inv(cobj.L.toarray())
print(c)
# plt.imshow(c, interpolation="nearest", cmap=plt.get_cmap("RdBu"))
# plt.savefig('C.jpg')

#plt.show()
plt.figure()

plt.spy(c)
plt.savefig('C_test.jpg')