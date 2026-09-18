# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the PSD target-averaging helpers in ``plot_utils``."""

import numpy as np

from weathergen.evaluate.plotting.plot_utils import (
    _average_target_psd,
    _compute_psd_gap_grid,
    _target_legend_label,
)


def _ds(freq, target, pred=None):
    return {
        "frequencies": np.asarray(freq, dtype=float),
        "psd_target": np.asarray(target, dtype=float),
        "psd_prediction": np.asarray(pred if pred is not None else target, dtype=float),
        "psd_method": "sht",
    }


def test_average_target_psd_two_runs():
    """The mean is taken over power, per frequency bin."""
    datasets = [_ds([1, 2, 3], [1.0, 2.0, 4.0]), _ds([1, 2, 3], [3.0, 4.0, 6.0])]

    freq, tar_mean, n_used = _average_target_psd(datasets)

    np.testing.assert_allclose(freq, [1, 2, 3])
    np.testing.assert_allclose(tar_mean, [2.0, 3.0, 5.0])
    assert n_used == 2


def test_average_target_psd_single_run_is_identity():
    """A one-run overlay must reproduce that run's target exactly."""
    datasets = [_ds([1, 2, 3], [1.0, 2.0, 4.0])]

    freq, tar_mean, n_used = _average_target_psd(datasets)

    np.testing.assert_allclose(freq, [1, 2, 3])
    np.testing.assert_allclose(tar_mean, [1.0, 2.0, 4.0])
    assert n_used == 1


def test_average_target_psd_drops_mismatched_axis(caplog):
    """Runs on a different frequency axis are excluded rather than crashing the plot."""
    datasets = [
        _ds([1, 2, 3], [1.0, 2.0, 4.0]),
        _ds([1, 2], [10.0, 10.0]),
        _ds([1, 2, 3], [3.0, 4.0, 6.0]),
    ]

    with caplog.at_level("WARNING"):
        freq, tar_mean, n_used = _average_target_psd(datasets, context="t_2m step 0")

    np.testing.assert_allclose(freq, [1, 2, 3])
    np.testing.assert_allclose(tar_mean, [2.0, 3.0, 5.0])
    assert n_used == 2
    assert "t_2m step 0" in caplog.text


def test_average_target_psd_ignores_nan():
    """NaN bins in one run do not poison the mean."""
    datasets = [_ds([1, 2], [np.nan, 2.0]), _ds([1, 2], [4.0, 6.0])]

    _, tar_mean, n_used = _average_target_psd(datasets)

    np.testing.assert_allclose(tar_mean, [4.0, 4.0])
    assert n_used == 2


def test_target_legend_label_states_averaging():
    """The legend must say when the target line is an average, and how many runs it spans."""
    assert _target_legend_label(1) == "Target"
    assert _target_legend_label(3) == "Target (mean of 3 runs)"


def test_average_target_psd_over_first_last_leaves():
    """First/last plot averages 2 leaves per run, so the run count is leaves // 2."""
    # Two runs, each contributing its first- and last-step target.
    leaves = [
        _ds([1, 2], [1.0, 1.0]),  # run A, first
        _ds([1, 2], [3.0, 3.0]),  # run A, last
        _ds([1, 2], [5.0, 5.0]),  # run B, first
        _ds([1, 2], [7.0, 7.0]),  # run B, last
    ]

    _, tar_avg, n_leaves = _average_target_psd(leaves)

    assert n_leaves == 4
    assert max(n_leaves // 2, 1) == 2
    np.testing.assert_allclose(tar_avg, [4.0, 4.0])

    # The cross-run average must not coincide with either run's own two-step mean (2.0 / 6.0).
    assert not np.allclose(tar_avg, [2.0, 2.0])
    assert not np.allclose(tar_avg, [6.0, 6.0])


def test_compute_psd_gap_grid_sign_and_shape():
    """Gap grid is log(pred) - log(target): positive = over-prediction."""
    freq = np.array([1.0, 2.0])
    per_fstep = {
        0: _ds(freq, [1.0, 1.0], pred=[np.e, 1.0]),  # over-predicts bin 0 by exactly 1 in log
        1: _ds(freq, [1.0, 1.0], pred=[1.0, 1.0 / np.e]),  # under-predicts bin 1
    }

    out_freq, fsteps, grid = _compute_psd_gap_grid(per_fstep, label="run_a", variable="t_2m")

    np.testing.assert_allclose(out_freq, freq)
    assert fsteps == [0, 1]
    assert grid.shape == (2, 2)
    np.testing.assert_allclose(grid[0], [1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(grid[1], [0.0, -1.0], atol=1e-12)
