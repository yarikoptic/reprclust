#!/usr/bin/env python

import sys
sys.path.insert(0, '/home/contematto/exp/clust/3rd/pypvclust')

ds_type = str(sys.argv[1])

import pvclust as pv
from mvpa2.suite import *

dss = h5load('posterior_{0}_rall_datasets.hdf5.gz'.format(ds_type))

surf_mesh_fn = '/data/attention/life/recon/as00/new_surf/'\
               'ico32_lh.inflated_alCoMmedial.asc'
surf_mesh = surf.from_any('/data/attention/life/recon/as00/new_surf/'
                          'ico32_lh.inflated_alCoMmedial.asc')
y_dim = surf_mesh.vertices[:,1]
indices = np.array(range(len(y_dim)))
indexed = np.hstack((indices.reshape(-1,1), y_dim.reshape(-1,1)))
sorter = indexed[indexed[:,1].argsort()]
posterior = sorter[:,0][:len(y_dim)/2]
selector = list(posterior.astype(np.int64))

# a bit slow but this is what we have now
connectivity = pv.build_connectivity_matrix(surf_mesh_fn,
                                            nodes=selector)

# set up scales
r = 1/9**np.linspace(-1, 1, 13)
nrep = 1000
ncores = 32

dss = [ds.T for ds in dss]

# make sure we have correlations to fisher transform
if np.max(dss[0]) > 1.:
    dss = [ds - 1. for ds in dss]

# fisher transform
dss = [np.arctanh(ds) for ds in dss]

np.random.seed(42)
print('Running bootstrap ward')
boot = pv.bootstrap_ward_parallel2(dss, connectivity=connectivity,
                                   fx_data=pv.bootstrap_subject_mean,
                                   nrep=nrep, r=r, ncores=ncores)
h5save('boot_{0}_{1}_fisher.hdf5.gz'.format(ds_type, nrep), boot)

print('Runnin scaleboot')
sb = pv.scaleboot_py(boot, ncores=ncores)

# save cluster pvalues
cluster_pvs = pv.extract_cluster_pvalues(sb)
h5save('cluster-pvs_{0}_{1}_fisher.hdf5.gz'.format(ds_type, nrep), cluster_pvs)
