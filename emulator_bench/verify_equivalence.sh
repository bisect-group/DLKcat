#!/usr/bin/env bash
# Checks that the DLKcat wrapper matches the original code. Usage: verify_equivalence.sh [--gpu | --negative-control]
# Expected env: mldb (torch, rdkit, pandas, pytest). The script does not create envs.
# Default: CPU tests on 64 train, 16 val and 16 test real rows (tests/fixtures), seed 666, 3 epochs.
# --gpu also runs the fp32 CUDA epoch test against the noise floor. --negative-control scales the wrapper weights by 1.05; that test must fail.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "Expected conda env: mldb"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
test_file=tests/test_original_equivalence.py
flags=(-q -rs -p no:cacheprovider --color=no)
if [[ ${1:-} == --negative-control ]]; then
    out=$(EQUIV_NEGATIVE_CONTROL=1 python -m pytest "$test_file" "${flags[@]}" -s -m gpu 2>&1) || true
    echo "$out"
    if grep -Eq "[0-9]+ passed" <<<"$out" || ! grep -Eq "[0-9]+ failed" <<<"$out"; then
        echo "FAIL: the GPU test did not fail under the negative control"
        exit 1
    fi
    echo "OK: the GPU test fails under the negative control"
    exit 0
fi
if [[ ${1:-} == --gpu ]]; then
    out=$(python -m pytest "$test_file" "${flags[@]}" -s 2>&1) || { echo "$out"; exit 1; }
else
    out=$(CUDA_VISIBLE_DEVICES="" python -m pytest "$test_file" "${flags[@]}" -m "not gpu" 2>&1) || { echo "$out"; exit 1; }
fi
echo "$out"
if grep -Eq "[0-9]+ skipped" <<<"$out"; then
    echo "FAIL: tests were skipped"
    exit 1
fi
echo "OK: wrapper matches the original code"
