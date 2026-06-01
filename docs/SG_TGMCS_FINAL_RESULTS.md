# SG-TGMCS Final Evaluation

Final metric: **SG-TGMCS**  
Machine-readable score field: `T_GMCS_static`

CSV:

`outputs/tgmcs_static_eval_combined/summary_tgmcs_static_by_dataset.csv`

## Metric

SG-TGMCS is a training-free geometric continuity score for rendered 3D morphing videos.

It uses only classical image processing:

- foreground mask
- silhouette contour
- skeleton
- radial shape signature
- Hu moments

It does not use RGB/color changes in the final score. Appearance is diagnostic only.

The final score is:

```text
T_GMCS_static = T_GMCS * F_static
```

where `F_static` penalizes static / duplicated / hold-frame behavior:

```text
freeze_excess = max(0, (freeze_ratio - 0.12) / (1 - 0.12))
repeat_excess = max(0, (repeated_frame_ratio - 0.05) / (1 - 0.05))
static_excess = max(freeze_excess, repeat_excess)
F_static = exp(-static_excess)
```

Higher is better.

## IMPUS vs Ours

| Method | Videos | Mean SG-TGMCS | Median | Max | Min | Mean freeze | Mean repeat |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ours / `outputs/3Dmorphing` | 34 | 0.2354 | 0.2335 | 0.4280 | 0.0633 | 0.1962 | 0.1100 |
| IMPUS | 3 | 0.0382 | 0.0413 | 0.0426 | 0.0306 | 0.7593 | 0.7297 |

## IMPUS Results

| Case | SG-TGMCS | T-GMCS | F_static | freeze_ratio | repeated_ratio |
|---|---:|---:|---:|---:|---:|
| `bull_demon_king_to_head_trellis` | 0.0426 | 0.0900 | 0.474 | 0.778 | 0.730 |
| `red_tree_to_bee_trellis` | 0.0413 | 0.0846 | 0.489 | 0.750 | 0.730 |
| `excavator_to_bulldozer_trellis` | 0.0306 | 0.0626 | 0.489 | 0.750 | 0.730 |

## Ours Top Results

| Case | SG-TGMCS | freeze_ratio | repeated_ratio |
|---|---:|---:|---:|
| `Pigsy+Sun_Wukong_no_ot` | 0.4280 | 0.106 | 0.000 |
| `Super_Big_Mech+typical_vehicle_pirate_ship_ot_modify_gate` | 0.4053 | 0.106 | 0.000 |
| `Super_Big_Mech+bee_no_conflict` | 0.3874 | 0.106 | 0.000 |
| `bee+Super_Big_Mech_base` | 0.3829 | 0.106 | 0.000 |
| `Godzilla+bee` | 0.3816 | 0.106 | 0.000 |

## Ablation Results

### Super Big Mech + Bee

| Variant | SG-TGMCS | F_static | freeze_ratio |
|---|---:|---:|---:|
| `no_conflict` | 0.3874 | 1.000 | 0.106 |
| `ot_modify_gate` | 0.3553 | 1.000 | 0.102 |
| `w_gate` | 0.3447 | 1.000 | 0.106 |
| `no_conflict_plus` | 0.3271 | 1.000 | 0.106 |
| `default` | 0.2551 | 0.991 | 0.128 |
| `base` | 0.2341 | 1.000 | 0.106 |

### Godzilla + Bee

| Variant | SG-TGMCS | F_static | freeze_ratio |
|---|---:|---:|---:|
| `default` | 0.3816 | 1.000 | 0.106 |
| `ot_modify_gate` | 0.3273 | 1.000 | 0.106 |
| `base` | 0.2329 | 1.000 | 0.106 |
| `CGAR_CA_OC` | 0.1184 | 0.603 | 0.542 |
| `CGAR` | 0.1026 | 0.603 | 0.542 |

### Chong + Hudie

| Variant | SG-TGMCS | F_static | freeze_ratio |
|---|---:|---:|---:|
| `50frames_CA_OC_MAVF_TOPO_REPAIR_CGAR_noMCRF` | 0.2983 | 1.000 | 0.106 |
| `ot_modify_gateno_sa` | 0.2284 | 1.000 | 0.106 |
| `ot_modify_gate` | 0.1500 | 1.000 | 0.106 |
| `modify_gateno_ot` | 0.1235 | 1.000 | 0.106 |
| `modify_gateno_sa` | 0.1222 | 1.000 | 0.106 |
| `CGAR_CA_OC_GOAVF_QFIELD_B_local_residual` | 0.0731 | 0.630 | 0.500 |
| `20frames_topo_repair_no_coord_cache` | 0.0684 | 0.581 | 0.578 |
| `CGAR_CA_OC_GOAVF_QFIELD_B_local_residual_arctan` | 0.0646 | 0.630 | 0.500 |
| `20frames_mavf_modify_gate_tfsa_cache_endpoint` | 0.0633 | 0.581 | 0.578 |

