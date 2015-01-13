from mvpa2.suite import *
import sys

sys.path.insert(0, '../../3rd/pypvclust')
import pvclust

nrep = 1000
ds_type = str(sys.argv[1])

boot_fn = 'boot_{0}_{1}.hdf5.gz'.format(ds_type, nrep)
cluster_pv_fn = 'cluster-pvs_{0}_{1}.hdf5.gz'.format(ds_type, nrep)

boot = h5load(boot_fn)
cluster_pv = h5load(cluster_pv_fn)

# stuff to get center_ids
surf_mesh = surf.from_any('/data/attention/life/recon/as00/'\
                          'new_surf/ico32_lh.inflated_alCoMmedial.asc')
y_dim = surf_mesh.vertices[:, 1]
indices = np.array(range(len(y_dim)))
indexed = np.hstack((indices.reshape(-1,1), y_dim.reshape(-1,1)))
sorter = indexed[indexed[:,1].argsort()]
posterior = sorter[:,0][:len(y_dim)/2]
selector = list(posterior.astype(np.int64))

s = 'as00'
anat_ds = h5load('/data/attention/mvpa/{0}/'\
                 'action-task_rall_r100_neural_'\
                 'sim_swrpN19_GAM_REML.hdf5.gz'.format(s))

center_ids = anat_ds.fa.center_ids
center_ids = center_ids[selector]

alphas = [0.05, 0.01, 0.001]

# largest first
for alpha in alphas:
    sig_clust = pvclust.significant_clusters(boot, cluster_pv, 
                                             alpha=alpha, largest=True)
    fn_out = 'cluster_pvclust_{0}_{1}_'\
             'largest.niml.dset'.format(ds_type, alpha)
    print('Saving {0}'.format(fn_out))
    pvclust.save_clusters(sig_clust, center_ids, fn_out)

# smallest then
for alpha in alphas:
    sig_clust = pvclust.significant_clusters(boot, cluster_pv, 
                                             alpha=alpha, largest=False)
    fn_out = 'cluster_pvclust_{0}_{1}_'\
             'smallest.niml.dset'.format(ds_type, alpha)
    print('Saving {0}'.format(fn_out))
    pvclust.save_clusters(sig_clust, center_ids, fn_out)
