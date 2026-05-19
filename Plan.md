# DLKcat EMULaToR Bench Plan

## 1. Baseline Summary From README

- Baseline repository: `/home/adhil/github/DLKcat`
- Task addressed by the baseline: in vitro `kcat` prediction from substrate SMILES and protein sequence.
- README and docs inspected: `README.rst`, `Code/model/preprocess_all.py`, `Code/model/run_model.py`, `Code/model/model.py`, and prediction examples.
- Recommended training command from docs: the README documents prediction and points model construction to `Code/model`; the train script accepts DLKcat hyperparameters as positional CLI arguments.
- Recommended evaluation command from docs: evaluation is performed during training on validation/test splits.
- Default config or hyperparameter source: paper settings and existing result names: radius 2, ngram 3, dim 20, GNN/CNN/output layers 3, window 11, learning rate `1e-3`, decay `0.5` every 10 epochs, weight decay `1e-6`, 50 epochs.
- Notes about doc or code mismatch: the original script randomly shuffles an already-featurized combined dataset into 80/10/10 splits; this bench uses explicit EMULaToR train/validation/test files while preserving the feature representation and model equations.

## 2. Expected Input Format

- Raw file format(s): parquet or CSV split files.
- Required columns or fields: `smiles`, `sequence`, `value`.
- Label or target fields: train internally on `log2(value)` to match DLKcat; report metrics on `log10(value)`, using `log10_value` when present for ground truth.
- Identifier fields: SMILES and normalized protein sequence hashes are used as cache keys.
- Assumptions about train, validation, and test partitioning: each split group contains `train`, `val`, and `test` files either directly or inside threshold subdirectories.

## 3. Featurization and Preprocessing Path

- Native preprocessing entrypoint: `Code/model/preprocess_all.py`.
- Native featurization functions: RDKit molecule parsing, atom/bond dictionaries, Weisfeiler-Lehman radius fingerprints, adjacency matrices, and protein n-gram splitting.
- Required intermediate artifacts: fingerprint/atom/bond/edge/word dictionaries plus molecule and protein feature caches.
- Where cached features will be stored: `/home/adhil/github/EMULaToR/data/processed/baselines/DLKcat/embeddings/dlkcat_radius{radius}_ngram{ngram}` by default.
- Cache format for train, validation, and test: one `.npz` per unique SMILES and sequence, plus split manifest CSV files that point rows to cache keys and targets.

## 4. Training and Evaluation Entrypoints

- Training entrypoint: `emulator_bench/train_single_target_tvt.py`.
- Evaluation entrypoint: final train/validation/test evaluation is run by the training entrypoint.
- Checkpoint location or discovery rule: `bestmodel.pth`, `bestmodel_state_dict.pth`, and `checkpoint_last.pt` inside each run directory.
- Default settings that must remain unchanged: paper feature settings and model architecture in `emulator_bench/default_hparams_paper.json`.
- Exact wrapper-to-baseline command mapping: the wrapper loads cached DLKcat features and trains a batched PyTorch 2.x implementation of the original GNN/CNN/attention equations.

## 5. Dataset and Split Mapping

- User dataset path: `/home/adhil/github/EMULaToR/data/processed/baselines/DLKcat`.
- User split definition path or files: `random_splits_grouped_sequence`, `random_splits_grouped_smiles`, `uniprot_time_splits`, `substrate_splits`, `enzyme_sequence_splits`, `enzyme_structure_splits`, and `conformer_cosine_splits`.
- Mapping from user schema to baseline schema: `smiles` maps to substrate graph, `sequence` maps to protein n-grams, and `value` maps to kcat.
- Mapping from user train, validation, and test splits to baseline expectations: explicit split files replace the original random split.
- Assumptions or blockers: 3D parquet files are not used by DLKcat.

## 6. Files To Add Under `emulator_bench/`

- `README.md`: bench overview, inputs, cache behavior, and commands.
- `commands.txt`: copy-ready commands.
- `default_hparams_paper.json`: paper/default retraining settings.
- `common.py`: shared path, split, metrics, JSON, and AMP helpers.
- `feature_pipeline.py`: DLKcat-compatible feature extraction and cache paths.
- `cache_embeddings.py`: one-time feature cache builder.
- `dataset.py`: cached dataset and collate function.
- `modeling.py`: batched DLKcat model.
- `train_single_target_tvt.py`: single split train/eval.
- `launch_parallel_retrain.py`: main paper-settings multi-GPU launcher.
- `tune_optuna.py`, `launch_parallel_optuna.py`, `launch_parallel_retrain_from_optuna.py`: optional Optuna workflow.
- `aggregate_dlkcat_results.py`: result aggregation.

## 7. Minimal Edits Outside `emulator_bench/`

- Required external edits: this `Plan.md` only.
- Why each edit is unavoidable: the adaptation workflow records the implementation plan at repo root.
- Why the edit does not change baseline behavior: it is documentation only.

## 8. Exact Execution Plan

1. Use `conda run -n mldb`.
2. Validate split schemas and discover train/validation/test jobs.
3. Build shared DLKcat feature dictionaries and one-time caches.
4. Train a selected split with paper settings using AMP and TF32 on CUDA.
5. Launch parallel retraining across GPUs from `default_hparams_paper.json`.
6. Optionally tune retraining-safe optimization parameters with Optuna.
7. Aggregate metrics and prediction outputs.
8. Run a small smoke test with `CUDA_VISIBLE_DEVICES=0`.
