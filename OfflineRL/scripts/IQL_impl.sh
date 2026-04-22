#!/bin/bash

echo "Starting the IQLNet training task..."

echo "Experiment 1: Fixed expectile with seed 42"
CUDA_VISIBLE_DEVICES=7 python -u IQL_impl.py -e 20 -b 256 -S 42 -t "fixed"

echo "Experiment 2: Fixed expectile with seed 123"
CUDA_VISIBLE_DEVICES=7 python -u IQL_impl.py -e 20 -b 256 -S 123 -t "fixed"

echo "Experiment 3: Fixed expectile with seed 999"
CUDA_VISIBLE_DEVICES=7 python -u IQL_impl.py -e 20 -b 256 -S 999 -t "fixed"

echo "Experiment 4: Linear expectile update with seed 42"
CUDA_VISIBLE_DEVICES=7 python -u IQL_impl.py -e 20 -b 256 -l True  -S 42 -t "linear"

echo "Experiment 5: Linear expectile update with seed 123"
CUDA_VISIBLE_DEVICES=7 python -u IQL_impl.py -e 20 -b 256 -l True  -S 123 -t "linear"

echo "Experiment 6: Linear expectile update with seed 999"
CUDA_VISIBLE_DEVICES=7 python -u IQL_impl.py -e 20 -b 256 -l True  -S 999 -t "linear"

echo "Experiment 7: Step expectile update with seed 42"
CUDA_VISIBLE_DEVICES=7 python -u IQL_impl.py -e 20 -b 256 -P True -S 42 -t "step"

echo "Experiment 8: Step expectile update with seed 123"
CUDA_VISIBLE_DEVICES=7 python -u IQL_impl.py -e 20 -b 256 -P True -S 123 -t "step"

echo "Experiment 9: Step expectile update with seed 999"
CUDA_VISIBLE_DEVICES=7 python -u IQL_impl.py -e 20 -b 256 -P True -S 999 -t "step"

echo "Finishing IQLNet training"

echo "Starting inference"
CUDA_VISIBLE_DEVICES=7 python -u IQL_inference.py


echo "Process finished at $(date)"