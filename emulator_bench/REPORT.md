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

## Wrapper
- `cache_embeddings.py` builds the feature caches with the original featurizer.
- `launch_parallel_retrain.py` finds 7 split groups on disk (random grouped x2, uniprot time, enzyme sequence, substrate, enzyme structure, conformer cosine) and runs every split with seeds 666 777 888. That is 21 runs.
- `train_single_target_tvt.py` trains on the original recipe, evaluates the last-epoch model, and writes metrics and predictions in log10 space.
- `aggregate_dlkcat_results.py` writes the mean and std over seeds.
- Val and test rows with unbonded fragments are featurized as in the DeepEnzyme wrapper (empty neighbour entry, unseen ids map to 0). Rows that still fail are dropped and counted in `run_summary.json`. The train filter is unchanged.
- `rescore_repaired_test.py` rescored the finished runs on the full test split with the saved last-epoch models.
- Dictionary order follows the original (full task table, then split-only items).

## Wrapper fixes in this pass
- `common.py`: added `enzyme_structure_splits` and `conformer_cosine_splits` (7 groups, 21 runs).
- `modeling.py`: unbatched `KcatPrediction` with the original equations (batch size 1, no masks).
- `dataset.py`: `PreloadedSplit` keeps all tensors on the GPU. The DataLoader and `--num_workers` are removed.
- `train_single_target_tvt.py`: no per-step host sync. The train loss is summed on the GPU. A run resumes from `checkpoint_last.pt` (model, optimizer, train order, NumPy random state). Batch size other than 1 is an error.
- `launch_parallel_retrain.py`: `--num_workers` removed.

## Equivalence with the original (tests/test_original_equivalence.py)
Data: 64/16/16 real kcat rows, seed 666, 3 epochs.
- CPU, fp32: features, dictionaries, initial weights, train order, per-epoch weights, final weights and test predictions are bitwise equal to the original `preprocess_all.py` and `run_model.py`. Resume gives the same weights as an uninterrupted run. Drop counts match.
- GPU, fp32 under CUDA MPS: one epoch stays inside the noise floor measured by perturbing the original at the unit roundoff (K = 10). The negative control (weights x 1.05) fails the test, as it must.
- Run: `bash emulator_bench/verify_equivalence.sh` (CPU) and `--gpu`.
- Local: 6 passed (CPU). Blackwell: GPU test passed and negative control failed.

## Speed-ups
Exact changes only. Same equations, same batch size 1, same fp32.
- Tensors preloaded on the GPU. No DataLoader, no per-step host sync.
- Several runs share one GPU under CUDA MPS. One process uses about half of the GPU time. Time slicing gives about 2x, MPS gives about 5.3x with 21 runs (random_splits_grouped_sequence, one epoch):
  - 1 run alone: 70 s per epoch.
  - 21 runs together under MPS: 277 s per epoch each, that is 13 s per run-epoch.
- Not used: bf16 `--amp` (94 s against 70 s per epoch at batch size 1).

## Precision
fp32, as the original. bf16 is slower and not used.

## Time and ETA (RTX PRO 4500 Blackwell)
- 277 s per epoch with 21 concurrent runs. 50 epochs give about 3.9 h. Larger splits may add some time. Plan 5 h.
- DLKcat and Theia can run at the same time. Both finish before Mon 2 Nov 05:00 IST.

## Launch
```
export CUDA_MPS_PIPE_DIRECTORY=$HOME/mps CUDA_MPS_LOG_DIRECTORY=$HOME/mps/log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY emulator_bench/launch_parallel_retrain.py --gpus 0 --runs_per_gpu 21
$PY emulator_bench/aggregate_dlkcat_results.py
```

## Open issues
- Batch size 1 and fp32 follow the released code. Old table rows used batch size 128 with bf16. The appendix text may need a DLKcat exception.
- The original has no validation checkpointing. The last epoch is final.
- The 50-epoch value comes from result file names in `Code/analysis`.
- Frozen dictionaries map unseen fingerprints of repaired rows to id 0, which the models never trained.
