#!/bin/env bash

if (command -v module &> /dev/null)
then
module load cmake gcc tbb cuda
fi

declare here="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export OMP_PROC_BIND=TRUE
export OMP_NUM_THREADS=20

set -x;

sbatch \
--account "" \
--mem-per-cpu=100000 \
--nodes=1 \
--ntasks=1 \
--cpus-per-task=20 \
--gres=gpu:4 \
--partition=gpu \
--time=24:00:00 ${here}/run.sh
