# Attaching a backbone decoder to a run that has none

A run trained with a latent-only loss carries encoder and forecast-engine weights but no decoder.
An *overlay config*, passed with `--config`, adds the missing pieces without touching the run's
stored config:

```bash
uv run inference --from-run-id <MODEL> \
  --config config/inference_decoder_overlay_<BACKBONE>.yml \
  --options <the usual inference options>
```

`scripts/make_decoder_overlay.py` writes that file. Read the four blocks below to check its output
or to write one by hand.

## Block 1 — the physical loss (always)

```yaml
validation_config:
  losses:
    physical:
      type: LossPhysical
      weight: 0.0                 # computed and logged, but not added to the total
      target_and_aux_calc: Physical
      loss_fcts:
        mse: {}
```

It must sit in `validation_config` (or `training_config`): that is what makes the decoder modules
get built at all. `test_config` inherits from `validation_config`, so this also satisfies the
"exactly one `LossPhysical` term" that the diagnostics and `write_output` require. Weight `0.0`
keeps the term out of the combined loss while still computing and logging it.

## Block 2 — the decoder weights (always)

```yaml
load_decoder_chkpt: {run_id: <BACKBONE>, mini_epoch: -1}
```

This overlays *only* `embed_target_coords.*`, `target_token_engines.*` and `pred_heads.*` on top of
the primary checkpoint; encoder and forecast engine are left alone. The backbone is normally the
run named in the model's `load_chkpt`:

```bash
python3 -c "
import json
print(json.load(open('models/<MODEL>/model_<MODEL>.json'))['load_chkpt'])
"
```

Confirm that checkpoint actually carries a decoder and read off its shapes:

```bash
uv run python -c "
import torch
sd = torch.load('models/<BACKBONE>/<BACKBONE>_latest.chkpt',
                map_location='meta', mmap=True, weights_only=True)
for k, v in sd.items():
    if k.startswith(('embed_target_coords', 'pred_heads')):
        print(k, tuple(v.shape))
"
```

Two lines come back, e.g. `embed_target_coords.<S>.linear.weight (512, C)` and
`pred_heads.<S>.pred_heads.0.0.weight (T, 512)`. They fix everything the stream block must match:

- **`<S>`** — the decoder is keyed by stream *name*. The overlay must provide a stream with exactly
  this name.
- **`T`** — number of target channels of that stream.
- **`C`** — input width of the target-coordinate embedding, which is `geoinfo_size + 105`
  (`get_targets_coords_size`: `geoinfo + 5*(3*5) + 3*8 + 6`). So `C` implies the number of geoinfo
  channels the stream must declare.

A mismatch in `T` or `C` is a hard `size mismatch` at load; a mismatch in the *name* is silent —
`load_state_dict(strict=False)` drops the weights and you decode with random ones.

## Block 3 — the decoded stream (when the model's own config lacks it)

Copy the stream **verbatim from the backbone's model JSON**; do not hand-write it:

```bash
python3 -c "
import json, yaml
d = json.load(open('models/<BACKBONE>/model_<BACKBONE>.json'))
print(yaml.safe_dump({'streams': {'<S>': d['streams']['<S>']}}, sort_keys=False))
"
```

Keep the derived `train_source_channels` / `train_target_channels` / `val_source_channels` /
`val_target_channels` / `target_channel_weights` lists that the JSON carries. They are computed
only when streams are read from a `streams_directory`, which does not happen for an overlay config
— and with `*_target_channels` missing, `is_stream_forcing` sees zero target channels, classifies
the stream as forcing, and builds no decoder for it at all, without an error.
`val_target_channels` is also the list that `diag_channels` names must come from.

Then switch off reconstruction for the model's own stream(s), so no second, randomly initialised
decoder is created:

```yaml
streams:
  <MODEL-STREAM>:
    reconstruct: false
```

Needed for any stream that declares target channels. Its geoinfo count typically differs from the
decoded stream's, so its decoder could not take the checkpoint weights in any case.

## Block 4 — the offset=1 inference alignment (always, inference only)

```yaml
test_config:
  model_input:
    forecasting:
      num_steps_input: 1
  forecast:
    offset: 1
    policy: "fixed"
```

Relabels the model's sole loaded input step as "now" and its forecast step(s) as starting +1 x
`time_step` ahead, so a diffusion model's RMSE rollout aligns with a deterministic (offset=1)
model's lead times. See the `fe_diffusion_model_conditioning="forecast"` branch in
`src/weathergen/model/model.py` (`stage=="inference" and tokens.shape[1]==1 and
forecast_offset==1`).

This block must live under `test_config` and nowhere else. `test_config` only applies during
inference (`inference` merges `validation_config` with `test_config`; `train`/`train_continue`
never read `test_config` at all), so putting it there is what keeps it out of training and
validation. Putting the same keys under `training_config` or `validation_config` instead would
wrongly relabel the input step during training/validation too — always add it under `test_config`,
never elsewhere. `make_decoder_overlay.py` writes this block unconditionally in every overlay it
generates, whether or not the model needs RMSE-vs-lead-time alignment for this particular run —
it's harmless to a plain physical-output overlay (a single input step is always "now" whether or
not you also care about matching a deterministic model's lead times), so there is no reason to
omit it. If you write an overlay by hand instead of using the script, add it too.

## Verifying before you burn GPU time

```bash
uv run python -c "
import weathergen.common.config as config
from pathlib import Path
from weathergen.utils.utils import is_stream_reconstructed
cf = config.load_merge_configs(None, '<MODEL>', -1, None,
                               Path('config/inference_decoder_overlay_<BACKBONE>.yml'), {})
print('streams:', list(cf.streams))
print('reconstructed:', {k: is_stream_reconstructed(v) for k, v in cf.streams.items()})
print('val losses:', {k: v.type for k, v in cf.validation_config.losses.items()})
print('load_decoder_chkpt:', cf.load_decoder_chkpt)
print('test_config.forecast:', cf.test_config.forecast)
print('test_config.model_input.forecasting:', cf.test_config.model_input.forecasting)
"
```

Expect exactly one stream reconstructed — the decoded one — one `LossPhysical` term, the
backbone in `load_decoder_chkpt`, and `test_config.forecast == {offset: 1, policy: fixed}` with
`test_config.model_input.forecasting.num_steps_input == 1` (Block 4). During the run, check the
log for `Loading decoder weights from id=...` and for any `Missing keys` naming decoder modules.

## Caveat to state in any write-up

A backbone decoder was trained on encoded ground-truth latents, never finetuned on
generated ones, so its error is part of every physical number produced this way. The
`decode(z)`-vs-truth panels of the ODE diagnostics are the control for exactly this.
