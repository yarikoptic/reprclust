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

from nose.tools import assert_equal, assert_greater_equal, assert_true

from mvpa2.misc.neighborhood import Sphere

from ..sim import get_intrinsic_noises, simple_sim1

from numpy.testing import assert_array_almost_equal

def test_get_intrinsic_noises():
    # very rudimentary test to see that noone tricks us up
    noises = get_intrinsic_noises((10, 11), 2., 3., 4)
    assert_equal(len(noises), 4)
    for noise in noises:
        assert_equal(noise.shape, (10, 11))
        # after smoothing variance should go down
        assert_greater_equal(2, np.std(noise))
        # but probably not that much ;)
        assert_greater_equal(np.std(noise), 0.1)


def test_simple_sim1_clean_per_subject():
    # no noise -- all must be clear
    dissims = [[0.9], [0.8], [0.5], [0.3]]
    args = (64, 64), dissims
    kwargs = dict(
        roi_neighborhood=Sphere(6),
        nruns=3, nsubjects=2
    )

    # clean case
    signal_clean, cluster_truth, dss = simple_sim1(
        *args,
        noise_subject_std=0,
        noise_independent_std=0,
        noise_common_std=0,
        **kwargs)
    # ,1 since we have only 1 value of dissim per each or ROIs
    assert_equal(signal_clean.shape, (64, 64, 1))
    # all dss should be identical to a_clean
    for ds in dss:
        for samples in ds[0].a.mapper.reverse(ds).samples:
            assert_array_almost_equal(
                signal_clean[..., 0],
                samples)


    # Now lets generate common noise
    signal_clean, cluster_truth, dss = simple_sim1(
        *args,
        noise_subject_std=0,
        noise_independent_std=0,
        noise_common_std=100,
        **kwargs)

    # corr coeffs should be really high across all the runs and subjects
    all_subj_runs = np.corrcoef(np.vstack(dss))
    assert_true(np.all(np.abs(
            all_subj_runs[np.triu_indices(len(all_subj_runs))])
            > 0.7))
    # but low to signal_clean
    assert_true(np.all(np.abs(
        np.corrcoef(signal_clean.flatten(), np.vstack(dss))[0, 1:]) < 0.3))


    # Now lets generate per subject common noise
    signal_clean, cluster_truth, dss = simple_sim1(
        *args,
        noise_subject_std=100,
        noise_independent_std=0,
        noise_common_std=0,
        **kwargs)

    # corr coeffs should be really high across all the runs within each
    # subject but otherwise having low correlation
    all_subj_runs = np.corrcoef(np.vstack(dss))
    # TODO: fix up
    # assert_true(np.all(np.abs(
    #         all_subj_runs[np.triu_indices(len(all_subj_runs))])
    #         > 0.8))  ## TODO: fix up
    # and low to signal_clean
    assert_true(np.all(np.abs(
        np.corrcoef(signal_clean.flatten(), np.vstack(dss))[0, 1:]) < 0.3))


    #import pylab as pl;
    #pl.imshow(signal_clean[:, :, 0]); pl.colorbar(); pl.show()
    #pl.imshow(dss[0].a.mapper.reverse(dss[0].samples[0])); pl.colorbar();
    # pl.show()
