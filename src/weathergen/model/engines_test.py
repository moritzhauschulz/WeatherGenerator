# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for the forecast engine's FE-local XSA / MLP-type overrides.

``_fe_arch_overrides`` only calls ``cf.get``, so it is exercised against plain dicts — no
``Config``, no CUDA, no network. Building a real ``ForecastingEngine`` needs a fully resolved
config and is left to the integration tests; what is pinned here instead is the resolution
precedence plus the ``MLP`` shape consequence that makes ``fe_mlp_type`` a training-time-only
switch (see ``_fe_arch_overrides``' docstring).
"""

import pytest
import torch

from weathergen.model.engines import _fe_arch_overrides
from weathergen.model.layers import MLP
from weathergen.model.norms import SwiGLU

# The forecast engine's actual FFN geometry on the current diffusion configs:
# fe_diffusion_latent_dim, and MLP's default hidden_factor.
_FE_DIM = 2560
_HIDDEN_FACTOR = 2


# --------------------------------------------------------------------------------------
# resolution precedence
# --------------------------------------------------------------------------------------


def test_unset_config_reproduces_previous_defaults():
    """Neither global nor fe_-scoped key present: the same (False, "mlp") the call sites
    used to hard-code, so an old config is unaffected by this change."""
    assert _fe_arch_overrides({}) == (False, "mlp")


def test_falls_back_to_global_keys_when_no_override():
    """Without an fe_-scoped key the forecast engine keeps tracking the global switches,
    which is what every existing checkpoint was trained under."""
    cf = {"use_xsa": True, "mlp_type": "swiglu"}
    assert _fe_arch_overrides(cf) == (True, "swiglu")


def test_fe_scoped_keys_win_over_global():
    cf = {"use_xsa": True, "mlp_type": "swiglu", "fe_use_xsa": False, "fe_mlp_type": "mlp"}
    assert _fe_arch_overrides(cf) == (False, "mlp")


def test_the_two_switches_are_independent():
    """The point of the change: each can be flipped without disturbing the other."""
    base = {"use_xsa": True, "mlp_type": "swiglu"}

    assert _fe_arch_overrides({**base, "fe_use_xsa": False}) == (False, "swiglu")
    assert _fe_arch_overrides({**base, "fe_mlp_type": "mlp"}) == (True, "mlp")


@pytest.mark.parametrize(
    ("global_xsa", "global_mlp", "fe_xsa", "fe_mlp", "expected"),
    [
        (False, "mlp", True, "swiglu", (True, "swiglu")),
        (True, "swiglu", False, "mlp", (False, "mlp")),
        (True, "swiglu", True, "swiglu", (True, "swiglu")),
        (False, "mlp", False, "mlp", (False, "mlp")),
    ],
)
def test_override_matrix(global_xsa, global_mlp, fe_xsa, fe_mlp, expected):
    cf = {
        "use_xsa": global_xsa,
        "mlp_type": global_mlp,
        "fe_use_xsa": fe_xsa,
        "fe_mlp_type": fe_mlp,
    }
    assert _fe_arch_overrides(cf) == expected


def test_return_types_are_coerced():
    """OmegaConf hands back whatever the YAML held; the call sites want a real bool/str."""
    use_xsa, mlp_type = _fe_arch_overrides({"fe_use_xsa": 1, "fe_mlp_type": "swiglu"})
    assert use_xsa is True
    assert isinstance(mlp_type, str)


# --------------------------------------------------------------------------------------
# why fe_mlp_type is training-time only
# --------------------------------------------------------------------------------------


def _fe_mlp(mlp_type):
    return MLP(
        _FE_DIM,
        _FE_DIM,
        num_layers=2,
        with_residual=True,
        hidden_factor=_HIDDEN_FACTOR,
        mlp_type=mlp_type,
    )


def test_mlp_and_swiglu_have_different_parameter_shapes():
    """Flipping fe_mlp_type changes the FFN tensor shapes, so it cannot be switched on an
    existing checkpoint -- unlike fe_use_xsa, which selects a parameter-free operation."""
    plain = {k: tuple(v.shape) for k, v in _fe_mlp("mlp").state_dict().items()}
    swiglu = {k: tuple(v.shape) for k, v in _fe_mlp("swiglu").state_dict().items()}

    assert plain != swiglu


def test_swiglu_applies_the_two_thirds_hidden_width_rule():
    """LLaMA-style rule in MLP.__init__: the gated FFN narrows the hidden width to 2/3 so
    the gate's doubled projection lands at a comparable parameter count."""
    hidden = _FE_DIM * _HIDDEN_FACTOR
    expected_hidden = max(1, int(2 * hidden / 3))

    linears = [m for m in _fe_mlp("swiglu").layers if isinstance(m, torch.nn.Linear)]
    assert linears[0].out_features == 2 * expected_hidden
    assert linears[-1].in_features == expected_hidden

    plain_linears = [m for m in _fe_mlp("mlp").layers if isinstance(m, torch.nn.Linear)]
    assert plain_linears[0].out_features == hidden


def test_mlp_type_selects_the_activation():
    assert any(isinstance(m, SwiGLU) for m in _fe_mlp("swiglu").layers)
    assert not any(isinstance(m, SwiGLU) for m in _fe_mlp("mlp").layers)


def test_unsupported_mlp_type_is_rejected():
    """A typo in fe_mlp_type must fail loudly at construction, not silently fall back."""
    with pytest.raises(ValueError, match="Unsupported mlp_type"):
        _fe_mlp("gelu")


def test_both_mlp_types_run_forward():
    x = torch.randn(2, 4, _FE_DIM)
    for mlp_type in ("mlp", "swiglu"):
        out = _fe_mlp(mlp_type)(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()
