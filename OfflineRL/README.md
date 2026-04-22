How to run: nohup ./IQL_impl.sh > training_log.out 2>&1 &

## Dataset names:
The dataset is preprocessed as:

Files:
- walker2d_train.npz
- walker2d_val.npz
- walker2d_test.npz

These files are generated using `dataset.py`. The training script `IQL_impl.py` assumes they already exist and will not regenerate them.