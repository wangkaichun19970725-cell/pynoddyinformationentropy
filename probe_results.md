# probe_results.md (template)

## Run Context
- Generator version under test: `<e.g., _gen_v10_3.py>`
- Noddy executable: `<path>`
- Probe command used: `<paste command>`
- Date/time:

## 1) Fold syntax and behavior mapping
- Accepted Fold fields observed in history excerpts:
  - `name`
  - `pos = [x,y,z]`
  - `amplitude`
  - `wavelength`
  - `axis_dir`
  - `plunge`
- Notes on any aliases / ignored keys:

### Visibility summary (from fold_trials.csv)
- Mean `complex_xz`:
- Mean `complex_yz`:
- Mean `yz_over_xz`:
- Best YZ trial index + params:
- Worst YZ trial index + params:

### Interpretation
- Parameter regions where YZ clearly shows folds:
- Parameter regions where only XZ shows folds:
- Suggested axis/plunge mixture policy:

## 2) Unconformity syntax and behavior mapping
- Accepted Unconformity fields observed in history excerpts:
  - `name`
  - `pos = [x,y,z_ref]`
  - `dip`
  - `dip_dir` / `dip_direction`
  - `num_layers`
  - `layer_thickness` / `layer_thicknesses`
- Notes on any aliases / ignored keys:

### Thickness + intersection summary (from unconf_trials.csv)
- Mean `unconf_top_proxy`:
- Max `unconf_top_proxy` trial + params:
- Trials where truncation appears clear in slices:

### Interpretation
- Parameter regions that keep unconformity thin but visible:
- Parameter regions causing domination of crop:
- Suggested `z_ref` / `dip` / `pos_xy` strategy to intersect folded package:

## 3) Minimal generator patch policy (fold + unconformity only)
- Fold policy:
  - Explicit axis_dir mixture: near-X / near-Y / oblique, with avoid-cardinal windows.
  - Randomized fold center (`pos`) per event.
- Unconformity policy:
  - Structural-focus placement (use mean of recent fold/fault points when available).
  - Cap total unconformity thickness to <= 20-25% of crop Z.
  - Keep unconformity event after folds/faults.

## 4) Quick pass/fail checklist
- [ ] YZ fold visibility improved (median `yz_over_xz` increased)
- [ ] Unconformity top proxy reduced (non-dominant)
- [ ] History excerpts confirm intended parameter fields
- [ ] No unrelated generator behavior changes
