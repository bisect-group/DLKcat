# DLKcat emulator_bench

This wrapper trains DLKcat with its original published settings on the default EMULaToR kcat splits. It runs several seeds per split. It does not tune hyperparameters. The files in `Code/` are not changed.

## Data

Default task directory: `data/processed/datasets/kinetic_dataset/kcat` (pass another directory with `--base_dir`).

The wrapper trains on every split folder it finds:

- `random_splits_grouped_sequence`
- `random_splits_grouped_smiles`
- `uniprot_time_splits`
- `enzyme_sequence_splits/threshold_<t>`
- `substrate_splits/threshold_<t>`

Each folder has `train`, `val` and `test` files (parquet or CSV) with the columns `sequence`, `smiles`, `value` and `log10_value`. The wrapper never writes into the data directory.

## Model and settings

DLKcat uses WL r-radius fingerprints of the hydrogen-expanded molecular graph and overlapping 3-gram protein words. A GNN encodes the molecule. An attention CNN encodes the protein. Output layers predict `log2(kcat)`.

Settings are in `default_hparams_paper.json`. They follow the released training script `Code/model/run_model.py`: Adam, lr 1e-3, lr x0.5 every 10 epochs, weight decay 1e-6, 50 epochs, batch size 1 (one update per sample), dim 20, radius 2, 3-gram, 3 GNN, CNN and output layers, window 11, fp32. The model after the last epoch is evaluated. The original has no early stopping and no best-checkpoint rule.

Training uses `log2(value)`. Metrics and predictions are in `log10` space. As in the original preprocessing, train rows with `.` in the SMILES or `value <= 0` are dropped. Val and test rows are kept if the model can featurize them. Rows that cannot be featurized are dropped, and `run_summary.json` (`rows_dropped`) counts them per split.

## Cache

Features are cached once per unique SMILES and sequence in `outputs/baselines/DLKcat/embeddings/dlkcat_radius2_ngram3` (change with `--embeddings_dir`). Per-split manifests point rows to the cached features.

Dictionary ids follow the original preprocessing. The original assigns atom, bond, fingerprint, edge and word ids in dataset row order, after it drops dotted SMILES and `value <= 0`. The wrapper does the same on the full, unsplit task table `<base_dir>/<task>_kinetic_params_1d.parquet` in stored row order. Items that only a split contains come after, sorted. The fingerprint dictionary depends on molecule order, so a different order gives a different table size and different initial weights. The cache folder name holds the cache version and a fingerprint of the table (`dlkcat_v2_radius2_ngram3_<fingerprint>`), so a changed table builds a new cache.

## Commands

See `commands.txt`. The main command is:

```bash
python emulator_bench/launch_parallel_retrain.py --gpus 0 1 2 3
```

It builds the cache, then runs every discovered split with seeds 666, 777 and 888. Finished runs are skipped unless `--overwrite` is set. Results go to `outputs/baselines/DLKcat/results/<split_group>/<split_name>/seed_<seed>` (change with `--output_root`).

Each run directory contains `model_state_dict.pth`, `logfile.csv`, `run_summary.json`, `final_results_{train,val,test}.csv` and `pred_label_{train,val,test}.csv`. `aggregate_dlkcat_results.py` writes mean and std over seeds.

## Batch order

The original shuffles the train list in place with `np.random.shuffle` at the start of every epoch, after the lr decay. The wrapper does the same on an index array that the train loader reads, after `set_seed` seeds NumPy. The same seed gives the same batch order as the original.

## Speed

Batch size 1 is slow: expect hours per run on one GPU. The only speed-ups are the feature cache and DataLoader workers. They do not change the model or the training settings. The cache keeps the original dictionary order (see Cache).
