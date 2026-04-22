## Dataset names:
The dataset is preprocessed as:

Files:
- walker2d_train.npz
- walker2d_val.npz
- walker2d_test.npz

These files are generated using `dataset.py`. The training script `IQL_impl.py` assumes they already exist and will not regenerate them.

## Implicit Q-Learning
The file `IQL.py` in the `RL` subpackage implements an Implicit Q-Learning (IQL) model along with its associated training pipeline. The model is designed for offline reinforcement learning and learns policies directly from pre-collected datasets without environment interaction.

A training script, `IQL_impl.py`, is provided in the `scripts` directory. This script trains the model and generates plots of training and validation loss (Actor, Q, and Value losses) across epochs. It also exports the best-performing model as an ONNX file for inference.

The shell script `IQL_impl.sh` runs a series of experiments with predefined hyperparameters, varying the expectile scheduling strategy (fixed, linear, and step-based) and random seeds. This allows for evaluation of both performance and robustness across runs.

Inference is performed using `IQL_inference.py`, which evaluates all trained models on the test dataset. This script generates box plots comparing performance across methods and produces summary CSV files containing evaluation metrics.

## Running the Experiments
1. Navigate to the project directory:
    ```bash
    cd OfflineRL
    source .venv/bin/activate
    ```
2. Install dependencies and build (if needed):
    ```bash
    uv sync
    uv build
    ```
3. Run the training experiments:
    ```bash
    cd scripts
    ./IQL_impl.sh
    ```

    Alternatively, you can run in the background with:
    ```bash
    cd scripts
    nohup ./IQL_impl.sh > training_log.out 2>&1 &
    ```
4. Results and Outputs:
    After execution, results will be saved in the `scripts/results` directory:
    - Training Plots:
        - Actor, Q, and Value loss curves
        - Expectile schedules
        - Format: method_seed#_best_actor.png
    - Trained Models:
        - Best model saved in ONNX format
        - Format: method_seed#_best_actor.onnx
    - Inference Outputs:
        - Box plots comparing methods (MSE and MAE)
        - Summary CSV files with aggregeted results across runs