# DLKcat emulator_bench

This bench adds EMULaToR split-driven retraining to DLKcat without changing the original `Code/` files.

The default dataset path is:

`/home/adhil/github/EMULaToR/data/processed/baselines/DLKcat`

The expected split files are parquet or CSV files with `smiles`, `sequence`, and `value`. When `log10_value` is present, it is used for reporting metrics.

## Model Inputs

DLKcat uses:

- Substrate SMILES converted by RDKit into a hydrogen-expanded molecular graph.
- Weisfeiler-Lehman r-radius molecular fingerprints from the original DLKcat preprocessing.
- Protein sequences split into overlapping amino-acid n-grams.
- kcat targets trained as `log2(value)`, matching the original DLKcat training script.

Metrics and prediction CSVs are reported in `log10(value)` space for EMULaToR comparison.

## Cache Layout

Feature caches are stored by default under:

`/home/adhil/github/EMULaToR/data/processed/baselines/DLKcat/embeddings/dlkcat_radius2_ngram3`

The cache contains:

- `dictionaries.pkl`: DLKcat atom, bond, edge, fingerprint, and protein word dictionaries.
- `molecules/<hash-prefix>/<hash>.npz`: one cached molecular fingerprint and adjacency per unique single-component SMILES.
- `proteins/<hash-prefix>/<hash>.npz`: one cached n-gram sequence array per unique normalized protein sequence.
- `split_manifests/.../{train,val,test}.csv`: row manifests pointing split rows to cached features and targets.
- `manifest.json`: cache metadata.

Repeated paper retraining, Optuna trials, and multi-GPU retraining reuse the same cache unless `--overwrite` is passed to `cache_embeddings.py`.

DLKcat’s original preprocessing excluded SMILES containing `.`. This bench preserves that behavior and records skipped rows in split manifest counts.

## Main Paper-Settings Workflow

Use conda env `mldb`.

```bash
conda run -n mldb python emulator_bench/cache_embeddings.py \
  --base_dir /home/adhil/github/EMULaToR/data/processed/baselines/DLKcat \
  --hparams_json emulator_bench/default_hparams_paper.json
```

```bash
conda run -n mldb python emulator_bench/launch_parallel_retrain.py \
  --gpus 0 1 2 3 \
  --base_dir /home/adhil/github/EMULaToR/data/processed/baselines/DLKcat \
  --hparams_json emulator_bench/default_hparams_paper.json \
  --seeds 666 667 668
```

Single split training:

```bash
conda run -n mldb python emulator_bench/train_single_target_tvt.py \
  --base_dir /home/adhil/github/EMULaToR/data/processed/baselines/DLKcat \
  --hparams_json emulator_bench/default_hparams_paper.json \
  --split_group random_splits_grouped_sequence \
  --device cuda:0
```

## Optional Optuna

Optuna tunes retraining-safe optimization/runtime parameters only. It does not tune the DLKcat architecture, feature settings, target transform, or split definitions.

```bash
conda run -n mldb python emulator_bench/tune_optuna.py \
  --base_dir /home/adhil/github/EMULaToR/data/processed/baselines/DLKcat \
  --split_groups random_splits_grouped_sequence random_splits_grouped_smiles \
  --device cuda:0 \
  --n_trials 20
```

## Outputs

Each run directory contains:

- `bestmodel.pth`
- `bestmodel_state_dict.pth`
- `checkpoint_last.pt`
- `logfile.csv`
- `run_summary.json`
- `final_results_train.csv`, `final_results_val.csv`, `final_results_test.csv`
- `pred_label_train.csv`, `pred_label_val.csv`, `pred_label_test.csv`

## Speed Enhancements

- One-time reusable DLKcat feature caches.
- Batched PyTorch 2.x implementation of the original DLKcat equations.
- Automatic mixed precision: bf16 on Ampere-or-newer CUDA devices, otherwise fp16 with `GradScaler`.
- TF32 enabled for CUDA matmul and cuDNN.
- Multi-GPU parallel retraining launcher using paper settings by default.
