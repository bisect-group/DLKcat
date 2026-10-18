# DLKcat emulator_bench report

## Method summary
DLKcat (Li et al., Nature Catalysis 2022) predicts kcat from a substrate SMILES and a protein sequence.
The molecule is a hydrogen-expanded RDKit graph. Atoms get WL r-radius fingerprint ids (r = 2).
A GNN (3 steps, mean pooling) gives the compound vector.
The protein is split into overlapping 3-grams. Embeddings pass through 3 CNN layers (window 11), then an attention step with the compound vector (mean pooling).
Both vectors are concatenated and pass through 3 output layers to one value: log2(kcat).
The loss is MSE on log2(kcat). The wrapper reports metrics in log10 space.
Task covered here: kcat only (ablations.tex lists DLKcat for kcat).

## Original default settings
| Setting | Value | Source |
|---|---|---|
| radius / n-gram | 2 / 3 | paper (Results, "selected optimal parameters"); Code/model/preprocess_all.py:113-114 |
| vector dim, GNN steps, CNN layers | 20, 3, 3 | paper; Code/analysis scripts use `dim20--layer_gnn3--layer_cnn3` |
| window, output layers | 11, 3 | paper (window 11 fixed); Code/model/model.py:51-53 |
| optimiser | Adam | Code/model/run_model.py:108-110 |
| lr, decay, interval | 1e-3, x0.5 every 10 epochs (applied at start of epochs 10, 20, ...) | Code/analysis/SuppleFig3a_five_times_validation.py filename; run_model.py:~230 |
| weight decay | 1e-6 | same filename |
| epochs | 50 | same filename (`iteration50`) |
| batch size | 1 (one Adam step per sample) | run_model.py `Trainer.train` loops over single samples |
| target | log2(kcat). Train filter: drop "." in SMILES and kcat <= 0 | preprocess_all.py:123,145 |
| precision | fp32 | no AMP in original |
| checkpointing | none: model saved every epoch, last epoch is final; no early stopping | run_model.py end of epoch loop |
| original split | random 80/10/10 with seed 1234 | run_model.py; paper |

Paper vs code: the paper does not state lr, epochs, batch size or optimiser. Values come from the code and the released result file names. The paper selects hyperparameters by learning curves on the validation set. The code has no checkpoint selection.

## What the wrapper does now
1. `cache_embeddings.py` builds per-SMILES and per-sequence feature caches (original featurisation) and per-split manifests.
2. `launch_parallel_retrain.py` discovers split folders on disk (random grouped x2, uniprot time, enzyme sequence threshold_*, substrate threshold_*), then runs each split x seed 666 777 888 in a queue over the given GPUs.
3. `train_single_target_tvt.py` trains with the settings above, logs val metrics each epoch, evaluates the last-epoch model on train/val/test, and writes metrics and predictions in log10 space.
4. `aggregate_dlkcat_results.py` writes mean/std over seeds.
Defaults: data `EMULaToR/data/processed/datasets/kinetic_dataset/kcat`; cache and results under `EMULaToR/outputs/baselines/DLKcat/{embeddings,results}`. Nothing is written under `data/`.

## Files removed or changed
Removed: `tune_optuna.py`, `launch_parallel_optuna.py`, `launch_parallel_retrain_from_optuna.py`, `run_split_benchmarks.py` (duplicate of the launcher, older single-GPU sweep).
Changed:
- `common.py`: Optuna constant removed; paths repo-relative; seeds default 666 777 888; five split groups (the two non-existent groups removed); AMP and metric-direction helpers removed.
- `default_hparams_paper.json`: batch_size 1; keys monitor_metric, val_every, patience, clip_grad removed.
- `train_single_target_tvt.py`: rewritten without AMP/TF32/cudnn.benchmark, best-val checkpointing, early stopping, clip_grad, and hyperparameter CLI overrides (only `--epochs` stays, for smoke tests). Default output dir is outside the data tree.
- `launch_parallel_retrain.py`, `cache_embeddings.py`, `aggregate_dlkcat_results.py`: new default paths, seeds, no extra flags; unused imports removed.
- `modeling.py`: unused import removed. `README.md`, `commands.txt` rewritten (no Optuna, no other user's paths).

## Bugs fixed
- Old trainer wrote results under the dataset split folders (`<split>/dlkcat_results`). Now outputs go to `--output_root`.
- Old default split groups included two folders that do not exist; thresholds are discovered from disk.
- `cudnn.benchmark=True` with variable shapes would re-tune per sample at batch size 1; removed.

## Deviations from the original protocol
- Data split: EMULaToR train/val/test files replace the random 80/10/10 split (required).
- Row filters: the original preprocessing filters (dot-SMILES, value <= 0) apply to train only, as in DEKP. Val and test keep every row the model can featurize. A val or test row is dropped only if the original featurizer cannot build its graph (RDKit parse failure, or a multi-component SMILES whose unbonded atoms break the WL step). `run_summary.json` lists `rows_dropped` per split. The numbers below are the dotted SMILES rows that fail featurization, per split folder on the files on disk (val/test): enzyme threshold_0.01 28/38 of 3056/3049, random sequence 31/20 of 3056/3057, random smiles 14/16, substrate threshold_0.205 36/136 of 3056/2686, uniprot time 54/28 of 3068/3034. No val or test row has value <= 0. So most dotted val/test rows are still dropped, because the original graph code cannot featurize them. The test sets stay about 1-5% smaller than the other baselines.
- Model code is a masked, batched re-implementation of the original equations. At batch size 1 there is no padding, so it is the same computation.
- Dictionary order: ids are assigned in row order of the full task table `<task>_kinetic_params_1d.parquet` (after the original dot and `value <= 0` filters), as the original does on its full dataset. Split-only items (for example dotted val/test SMILES) come after, sorted, so they can add ids the original never has. The cache folder name has the cache version (2) and a table fingerprint.
- Batch order: the train order comes from `np.random.shuffle` each epoch on one persistent index array, as in the original `Trainer.train`, not from a torch `DataLoader(shuffle=True)`.
- DataLoader workers and the feature cache change speed only.
- Val metrics are computed each epoch for the log only. They do not select the model. The original tested dev and test each epoch; the wrapper evaluates test only once at the end.
- Test-only flags: `--epochs`, `--sample_rows`, `--max_jobs`.

## Changes after review

- `feature_pipeline.py`, `cache_embeddings.py`: the dot-SMILES and value <= 0 filters now apply to train only. `featurize_smiles` no longer rejects "."; it raises only if the WL fingerprints do not cover all atoms. Val/test rows that cannot be featurized are dropped and counted (`split_manifest_index.csv`, `run_summary.json`).
- `cache_embeddings.py`: `dictionaries.pkl` is saved atomically after each batch of 2000 items, before the batch files are written. Every file on disk therefore has ids that the saved dictionary holds. After a crash the cache resumes without silent id drift. `--overwrite` removes the old manifest first, so a half-rebuilt cache is not used.
- `train_single_target_tvt.py`: `logfile.csv` is removed at the start of a run, so a restart writes one row per epoch. The run has no mid-run resume; it starts again from epoch 1.
- `launch_parallel_retrain.py`: exits with status 1 and lists the failed runs if any run fails (the summary is still written first).
- Output root moved from `data/processed/baselines/DLKcat` to `outputs/baselines/DLKcat`, as for MPEK and CatPred. `commands.txt` no longer has a home path (`$PY`).
- Smoke after the changes (mldb, GPU 1): cache for random_splits_grouped_sequence and uniprot_time_splits (3000 rows per file), then the launcher on uniprot_time_splits and substrate_splits, 1 epoch. The uniprot run finished with `rows_dropped` train 20, val 54, test 28; the substrate run failed because its cache did not exist, and the launcher exited 1 with the error. Output: `scratchpad/bench_smoke/DLKcat/r2`.
- Not changed: batch size 1 runtime (about 8-9 h per run).
- `cache_embeddings.py`: `split_manifest_index.csv` is merged across cache runs (rows keyed by `manifest_path`, the newest wins) instead of overwritten.
- `train_single_target_tvt.py`: `rows_dropped` is read at startup. A split missing from the index raises `KeyError` before training and names the missing manifests.
- Smoke (mldb, CPU, 64 rows): two cache runs with different split groups gave one index with both (6 rows + header); training a cached split finished; training an uncached split failed at once with the `KeyError`. Output: `scratchpad/bench_smoke/DLKcat`.

## Commands (all seeds x splits, kcat)
```
PY=/home/adhil/.conda/envs/mldb/bin/python
cd /home/adhil/github/EMULaToR/baselines/DLKcat
$PY emulator_bench/launch_parallel_retrain.py --gpus 0 1 2 3 --runs_per_gpu 1
$PY emulator_bench/aggregate_dlkcat_results.py
```
Splits and seeds default to all discovered folders and 666 777 888. Add `--base_dir <task dir>` for another dataset.

## Smoke test
Env: mldb, GPU 1 only, 64 rows per split, 1 epoch, seed 666, splits random_splits_grouped_sequence, random_splits_grouped_smiles, uniprot_time_splits, enzyme_sequence_splits/threshold_0.01, substrate_splits (cache for all five; training on several of them). All runs finished and wrote metrics, predictions and summary CSVs. Aggregation worked. Outputs: `/tmp/claude-1005/-home-adhil-github/92fb41ec-b482-467d-940d-08ea9f435509/scratchpad/bench_smoke/DLKcat`. Nothing was written under EMULaToR/data. Real training was not run.
Timing (1000 train + 1000 val rows, GPU, batch size 1): about 23 s per epoch, so about 10 min per epoch and about 8-9 h per run for 55k rows x 50 epochs. 15 runs need about 130 GPU hours. CPU is about 8x slower.

## Open issues
- Batch size 1 follows the released code but is slow. The earlier wrapper used batch size 128 with bf16 and best-val checkpointing. Results will differ from the table in ablations.tex, which were produced with those settings.
- camera_ready.tex says kinetic baselines use bf16. DLKcat now runs fp32 as the original did. The appendix text may need a DLKcat exception.
- Last-epoch checkpoint (no validation selection) follows the original code. The paper text says "validation-based checkpointing is kept when the original has it". The original has none.
- Enzyme-structure and conformer splits do not exist on disk, so P-struct and Mol-3D have no DLKcat runs.
- The five-fold files in the original repo were removed upstream, so the 50-epoch value comes from result file names in Code/analysis.
- Dotted-SMILES val/test rows are dropped as not featurizable (counted in `rows_dropped`). The launcher skips a finished run only when its `run_summary.json` epochs match the requested epochs; otherwise it reruns the run.

## Equivalence check against the original (kcat, 64/16/16 rows, seed 666, 3 epochs, CPU)
Wrapper CLI (`cache_embeddings.py`, `train_single_target_tvt.py`) against the original `preprocess_all.py` and `run_model.py` functions, on the same rows. Scratch table = train, val, test rows in the original order. Results: all five dictionaries identical (fingerprint 715, word 7369, atom 11, bond 4, edge 1463), initial weights max abs diff 0.0, batch order equal for all 3 epochs, final weights max abs diff 0.0, per-epoch val MAE/RMSE/R2 within 2e-8, train loss (mean x 64 against the original sum) within 4e-7 relative, final predictions within 8e-8 (log10). The original needed two NumPy 2 shims (ragged `np.save`, `math.pow` on size-1 arrays).
