#emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*- 
#ex: set sts=4 ts=4 sw=4 noet:
"""Few functions to simulate data for testing various clustering approaches

 COPYRIGHT: Uberclust gig 2015

 LICENSE: MIT

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
"""

import numpy as np
import skimage.filter
from scipy.spatial.distance import squareform, pdist
from mvpa2.base.dataset import vstack as dsvstack
from mvpa2.mappers.fx import mean_group_sample
from mvpa2.datasets.base import Dataset
from mvpa2.mappers.flatten import FlattenMapper
from mvpa2.misc.neighborhood import Sphere

from mvpa2.misc.fx import get_random_rotation

from .mds import mds_classical
import pylab as pl
# Target sample dissimilarities to "simulate"
DISSIM6 = []

def vector_form(m):
    m = np.asanyarray(m)
    if m.ndim > 1:
        return m[np.triu_indices(m.shape[0], 1)]
    return m


# TODO: while performing filtering, we are probably changing amount of
# variance, so we might want to "re-standardize" whenever smoothing noise

def filter_each_2d(d, sigma):
    if d.ndim == 3:
        # rollaxis is to move "vstacked" last dimension back to its place
        return np.rollaxis(
                 np.array([
                       skimage.filter.gaussian_filter(d[..., i], sigma)
                       for i in xrange(d.shape[2])]),
                 0, 3)
    elif d.ndim == 2:
        return skimage.filter.gaussian_filter(d, sigma)
    else:
        raise ValueError("Didn't yet bother for ndim > 3")


def get_intrinsic_noises(shape, std, sigma, n=1):
    """

    Parameters
    ----------
    shape:
       shape of the data
    n: int
       number of intrinsic noise components. >1 makes sense since we color them
       (otherwise mix of normals would be unmixable normal distribution) but yet
       to verify ;)
    std: float
       standard deviation in "time" (among samples)
    sigma: float
       spatial smoothing
    """
    # TODO: hold on
    return [filter_each_2d(np.random.normal(size=shape)*std, sigma)
            for i in xrange(n)]


def generate_mixins(npoints):
    """Generate random mixins with some coloring and offset"""
    data = np.random.normal(size=npoints)
    assert(len(data) == npoints)
    from scipy.stats import norm
    x = np.arange(npoints)
    kernel = norm.pdf(x/5. - 1.)
    data_filtered = np.convolve(data, kernel)[:len(data)] + np.random.normal()
    return data_filtered


def get_pattern_from_dissim(dissim, ndim=None, metric="euclidean"):
    """Generate a pattern fulfilling original dissimilarity

    Parameters
    ----------
    dissim : np.ndarray
      Dissimilarity
    ndim : int, optional
      target dimensionality
    metric : { 'euclidean' } # OTHERS TODO
    """
    dissim = np.asanyarray(dissim)
    if metric == 'euclidean':
        # Generate a pattern which would carry necessary similarity
        mds_pattern = mds_classical(dissim, ndim=-1)
        if not ndim:
            ndim = mds_pattern.shape[1]
        else:
            # we might need to subselect necessary number of leading dimensions
            mds_pattern = mds_pattern[:, :ndim]
        # and rotate it within the ROI
        ptrns = np.dot(mds_pattern,
                       get_random_rotation(mds_pattern.shape[1], ndim))
        return np.atleast_2d(ptrns)
    else:
        raise NotImplementedError("Don't know yet how to generate for %r metric"
                                  % metric)


def simple_sim1(shape, dissims,
                rois_arrangement='circle',
                roi_neighborhood=Sphere(5),
                dissim_metric='euclidean',
                nruns=1, nsubjects=1, targets=None,
                # noise components -- we just add normal for now also with
                # spatial smoothing to possibly create difference in noise
                # characteristics across different kinds
                #
                # "Instrumental noise" -- generic nuisance
                noise_independent_std=0.4, noise_independent_smooth=3.,
                # "Intrinsic signal", specific per each subject (due to
                # motion, whatever) -- might be fun for someone to cluster,
                # but irrelevant for us
                noise_subject_n=1, noise_subject_std=0.4, noise_subject_smooth=1.5,
                # "Intrinsic common signal" -- probably generalizes across
                # subjects and fun for someone studying veins to get those
                # reproducible clusters.  It will be mixed in also with
                # different weights per each run.
                # Again -- might be fun for someone to cluster, but not for us
                # since it would not be representative of the original signal
                noise_common_arrangement='stripes',
                noise_common_n=3, noise_common_std=0.4, noise_common_smooth=2.
                ):
    """Simulate "data" based on dissimilarity/clusters structure

    Simulated data (multiple runs per subject) will contain spatial patterns which
    in each cluster has target similarity structure (per each trial), and 3 noise
    components:

    - random normal noise, also spatially smoothed (should have smaller
      sigma for smoothing probably than for intrinsic noise)

    - intrinsic noise which is composed from a set of random fields,
      generated by random normal noise with subsequent spatial filtering,
      which are then mixed into each run data with random weights.  They
      are to simulate subject-specific intrinsic signals such as artifacts
      due to motion, possible subject-specific physiological processes

    - intrinsic common noise across subjects intrinsic noise (e.g. all of them
      have similar blood distribution networks and other physiological
      parameters, and some intrinsic networks, which although similar in
      space would have different mix-in coefficients across subject/runs)

    Theoretically, decomposition methods (such as ICA, PCA, etc) should help to
    identify such common noise components and filter them out.  Also methods
    which iteratively remove non-informative projections (such as GLMdenoise)
    should be effective to identify those mix-ins

    TODO: now mix-in happens with purely normal random weights,  ideally we
    should color those as well

    Parameters
    ----------

    shape : tuple (int, int)
      shape of 2d "voxel space"
    dissims : list of ndarray
      list of dissimilarities per each cluster/roi. Dissimilarities could be presented
      as vectors (upper triang elements) or full matrices of dissimilarities
    rois_arrangement : {'circle',}
      how to arrange clusters
    roi_neighborhood :
      neighborhood defining clusters
    nruns, nsubjects : int
    targets : list of string, optional
      Targets to assign (corresponding to dissims)
    noise_independent_std, noise_independent_smooth:
      characteristics of noise independent of each subject/sample
      (i.e. Instrumental noise like)
    noise_subject_n, noise_subject_std, noise_subject_smooth :
      characteristics of "Intrinsic signal", specific per each subject (due
      to motion, whatever) -- might be fun for someone to cluster,
      but irrelevant for us
    noise_common_n, noise_common_std, noise_common_smooth:
      characteristics of "Intrinsic common signal" -- probably generalizes across
      subjects and fun for someone studying veins to get those
      reproducible clusters.  Assumption is that cluster structure is common
      across subjects, but dissimilarity structure between targets of interest
      will be different per each run/subject, thus this signal overall will not
      correlate with the signal of interest.
      Again -- might be fun for someone to cluster, but not for us
      since it would not be representative of the original signal

    Returns
    -------
    signal_clean, cluster_truth, dss
    """

    for i, dissim in enumerate(dissims):
        dissim = np.asanyarray(dissim)
        if dissim.ndim == 1:
            dissim = squareform(dissim)
        dissims[i] = dissim  # replace them all with square version
        assert(dissim.shape[0] == dissim.shape[1])
    dissims = np.asanyarray(dissims)
    nrois = len(dissims)      # number of ROIs
    ncats = dissims.shape[1]  # number of categories

    if targets is None:
        targets = ['c%d' % i for i in xrange(ncats)]
    assert(len(targets) == ncats)

    # generate target clean "picture" per each subject/run
    # ncats x shape
    #d = np.asanyarray(dissims[0])
    signal_clean = np.zeros((ncats,) + shape)

    # generate ground truth for clustering
    cluster_truth = np.zeros(shape, dtype='int')

    if rois_arrangement == 'circle':
        radius = min(shape[:2])/4.
        center = np.array((radius*2,) * len(shape)).astype(int)
        # arrange at quarter distance from center
        for i, dissim in enumerate(dissims):
            # that is kinda boring -- the same dissimilarity to each
            # voxel???
            #
            # TODO: come up with a better arrangement/idea, e.g. to
            # generate an MVPA pattern which would satisfy the
            # dissimilarity (not exactly but at least close).  That
            # would make more sense
            roi_center = center.copy()
            roi_center[0] += int(radius * np.cos(2*np.pi*i/nrois))
            roi_center[1] += int(radius * np.sin(2*np.pi*i/nrois))
            roi_coords = roi_neighborhood(roi_center)

            # filter those out which are outside of our space
            def in_box(coords):
                acoords = np.asanyarray(coords)
                return np.all(acoords >= [0]*len(coords)) and \
                       np.all(acoords < signal_clean.shape[1:])

            roi_coords = filter(in_box, roi_coords)
            npoints = len(roi_coords)

            pattern = get_pattern_from_dissim(
                dissim, ndim=npoints, metric=dissim_metric)
            # TODO  tricky part here -- I thought to normalize the "signal" pattern
            # for the unit variance so noise std would be easier to comprehend
            # But in case of euclidean distance we shouldn't change (by default) the
            # scale anyhow if we want to really have it.  BUT that somewhat ruins
            # balancing between ROIs of different sizes etc

            # TODO: do proper numpy way

            for i, coords in enumerate(roi_coords):
                # for indexing within array
                acoords = np.asanyarray(coords)
                for signal_clean_, pattern_ in zip(signal_clean, pattern):
                    signal_clean_.__setitem__(acoords, pattern_[i])
                cluster_truth.__setitem__(acoords, i+1)
    else:
        raise ValueError("I know only circle")

    # # generated randomly and will be mixed into subjects with different weights
    # # TODO: static across runs within subject??  if so -- would be no different
    # #       from having RSAs?
    # common_noises = get_intrinsic_noises(
    #     signal_clean.shape,
    #     std=noise_common_std,
    #     sigma=noise_common_smooth,
    #     n=noise_common_n)
    # assert common_noises[0].ndim == 3, "There should be no time comp"
    #
    # we will have common noise being present as another set of parcels
    # which will have as well consistent structure across subjects BUT
    # will have random data (constant offset within each parcel/target
    # /chunk across all voxels, so valid only for "euclidean" distance
    # case ATM
    if noise_common_arrangement == 'stripes':
        # let's just do vertical stripes field
        cluster_common_noise = np.zeros(shape=shape, dtype=int)
        cluster_common_noise[:] = (np.arange(shape[0], dtype=float)*noise_common_n/shape[0]).astype(int)
    else:
        raise ValueError("Unknown arrangement for common noise clusters")

    # Now lets generate per subject and per run data by adding some noise(s)
    # all_signals = []
    dss = []
    for isubject in xrange(nsubjects):
        # Interesting noise, simulating some underlying process which has nothing
        # to do with original design/similarity but having spatial structure which
        # repeats through runs with random weights (consider it to be a principal component)

        # generated randomly for each subject separately, but they should have
        # common structure across runs
        subj_specific_noises = get_intrinsic_noises(signal_clean.shape,
                                                std=noise_subject_std,
                                                sigma=noise_subject_smooth,
                                                n=noise_subject_n)
        assert subj_specific_noises[0].ndim == 3, "There should be no time comp"
        # subject_signals = []
        dss_subject = []
        #subj_common_noises = [noise * np.random.normal()
        #                      for noise in common_noises]

        subj_specific_mixins = generate_mixins(nruns)

        for run in range(nruns):
            signal_run = signal_clean.copy()
            for noise in subj_specific_noises:
                signal_run += noise * subj_specific_mixins[run]
            for inoise in xrange(noise_common_n):
                signal_run[cluster_common_noise == inoise] += generate_mixins(ncats)
            # generic noise -- no common structure across subjects/runs
            signal_run += filter_each_2d(
                np.random.normal(size=signal_clean.shape)*noise_independent_std,
                noise_independent_smooth)

            ## rollaxis to bring similarities into leading dimension
            ds = Dataset(signal_run) # np.rollaxis(signal_run, 2, 0))
            ds.sa['chunks'] = [run]
            ds.sa['targets'] = targets
            # what did I have in mind???
            #ds.sa['dissimilarity'] = np.arange(len(sim))  # Lame one for now
            ds_flat = ds.get_mapped(FlattenMapper(shape=ds.shape[1:],
                                                  space='pixel_indices'))
            dss_subject.append(ds_flat)
            #subject_signals.append(signal_run)
        #all_signals.append(subject_signals)
        ds = dsvstack(dss_subject)
        ds.a['mapper'] = dss_subject[0].a.mapper   # .a are not transferred by vstack
        dss.append(ds)

    # Instrumental noise -- the most banal
    assert(len(dss) == nsubjects)
    assert(len(dss[0]) == nruns*len(dissim))

    return signal_clean, cluster_truth, dss


if __name__ == '__main__':
    a_clean, cluster_truth, dss = simple_sim1(
        (64, 64), [[1], [0.8], [0.5], [0.3]],
        roi_neighborhood=Sphere(6),
        nruns=3, nsubjects=2,
        noise_subject_n=1, noise_subject_std=5, noise_subject_smooth=5,
        noise_independent_std=4, noise_independent_smooth=1.5,
        noise_common_n=1, noise_common_std=3)

    # just a little helper
    def get2d(ds):
        return dss[0].a.mapper.reverse(ds)

    import pylab as pl
    pl.clf()
    DS = dsvstack(dss)
    # Sample plots
    for s in [0, 1]:
        ds2 = get2d(dss[0])
        for r in [0, 1]:
            pl.subplot(3,3,1+r+s*3); pl.imshow(ds2[ds2.sa.chunks == r].samples[0], interpolation='nearest'); pl.ylabel('subj%d' % s);  pl.xlabel('run1');
        pl.subplot(3,3,3+s*3); pl.imshow(get2d(mean_group_sample(['dissimilarity'])(dss[0]).samples)[0], interpolation='nearest'); pl.xlabel('mean');

    ds = dsvstack(dss)
    ds.a['mapper'] = dss[0].a.mapper
    ds_mean = mean_group_sample(['dissimilarity', 'chunks'])(ds)
    for r in [0, 1]:
        ds_mean_run0 = ds.a.mapper.reverse(ds_mean[ds_mean.chunks == r])
        pl.subplot(3,3,1+r+2*3); pl.imshow(ds_mean_run0.samples[0], interpolation='nearest'); pl.ylabel('mean(subj)');  pl.xlabel('run%d' % r)
    ds_global_mean = mean_group_sample(['dissimilarity'])(ds)
    pl.subplot(3,3,3+2*3); pl.imshow(get2d(ds_global_mean).samples[0], interpolation='nearest'); pl.xlabel('mean');

pl.show()
