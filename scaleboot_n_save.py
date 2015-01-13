#!/usr/bin/env python

import sys
sys.path.insert(0, '/home/contematto/exp/clust/3rd/pypvclust')

ds_type = str(sys.argv[1])

import pvclust as pv
from mvpa2.suite import *

nrep = 1000
ncores = 32

boot = h5load('boot_{0}_{1}.hdf5.gz'.format(ds_type, nrep))

print('Runnin scaleboot')
sb = pv.scaleboot_py(boot, ncores=ncores)
cluster_pvs = pv.extract_cluster_pvalues(sb)

h5save('cluster-pvs_{0}_{1}.hdf5.gz'.format(ds_type, nrep), cluster_pvs)
