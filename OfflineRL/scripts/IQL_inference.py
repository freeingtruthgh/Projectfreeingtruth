import os
import re
import csv
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt


class WalkerDataset:
    def __init__(self, path):
        data = np.load(path)
        self.observations = data["observations"].astype(np.float32)
        self.actions = data["actions"].astype(np.float32)


def run_batched_inference(session, observations, batch_size=1024):
    input_name = session.get_inputs()[0].name
    predictions = []

    for start_idx in range(0, len(observations), batch_size):
        end_idx = min(start_idx + batch_size, len(observations))
        obs_batch = observations[start_idx:end_idx]
        pred_batch = session.run(None, {input_name: obs_batch})[0]
        predictions.append(pred_batch)

    return np.concatenate(predictions, axis=0)


def evaluate_onnx_model(model_path, dataset_path, batch_size=1024):
    dataset = WalkerDataset(dataset_path)
    session = ort.InferenceSession(model_path)

    predicted_actions = run_batched_inference(
        session=session,
        observations=dataset.observations,
        batch_size=batch_size
    )

    mse = np.mean((predicted_actions - dataset.actions) ** 2)
    mae = np.mean(np.abs(predicted_actions - dataset.actions))

    return mse, mae


def parse_model_filename(filename):
    # Expected format: Method_seed123_best_actor.onnx
    pattern = r"^(?P<method>.+)_seed(?P<seed>\d+)_best_actor\.onnx$"
    match = re.match(pattern, filename)
    if match is None:
        return None

    return {
        "method": match.group("method"),
        "seed": int(match.group("seed"))
    }


def compute_confidence_interval(values):
    values = np.array(values, dtype=np.float64)
    mean = np.mean(values)
    std = np.std(values, ddof=1) if len(values) > 1 else 0.0
    ci = 1.96 * std / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, std, ci


def save_run_results_csv(results, output_path):
    with open(output_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["method", "seed", "test_mse", "test_mae"])

        for row in results:
            writer.writerow([row["method"], row["seed"], row["test_mse"], row["test_mae"]])

    print(f"Saved run results CSV to {output_path}")


def save_summary_csv(results, output_path):
    grouped = {}
    for row in results:
        method = row["method"]
        if method not in grouped:
            grouped[method] = {"mse": [], "mae": []}

        grouped[method]["mse"].append(row["test_mse"])
        grouped[method]["mae"].append(row["test_mae"])

    with open(output_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["method", "metric", "mean", "std", "ci_95"])

        for method, metrics in grouped.items():
            mse_mean, mse_std, mse_ci = compute_confidence_interval(metrics["mse"])
            mae_mean, mae_std, mae_ci = compute_confidence_interval(metrics["mae"])

            writer.writerow([method, "test_mse", mse_mean, mse_std, mse_ci])
            writer.writerow([method, "test_mae", mae_mean, mae_std, mae_ci])

    print(f"Saved summary CSV to {output_path}")


def plot_metric_boxplot(results, metric_key, ylabel, save_path):
    grouped = {}
    for row in results:
        method = row["method"]
        if method not in grouped:
            grouped[method] = []
        grouped[method].append(row[metric_key])

    labels = sorted(grouped.keys())
    data = [grouped[label] for label in labels]

    plt.figure()
    plt.boxplot(data, tick_labels=labels)
    plt.title(f"{ylabel} Across Runs")
    plt.xlabel("Method")
    plt.ylabel(ylabel)
    plt.savefig(save_path)
    plt.close()

    print(f"Saved boxplot to {save_path}")


def main():
    RESULTS_DIR = "results"
    BASE_DATA_DIR = "/home/tbl0009/data"
    test_path = os.path.join(BASE_DATA_DIR, "walker2d_test.npz")

    all_results = []

    for filename in os.listdir(RESULTS_DIR):
        if not filename.endswith("_best_actor.onnx"):
            continue

        parsed = parse_model_filename(filename)
        if parsed is None:
            continue

        model_path = os.path.join(RESULTS_DIR, filename)

        print(f"Evaluating {filename}")
        test_mse, test_mae = evaluate_onnx_model(model_path, test_path)

        all_results.append({
            "method": parsed["method"],
            "seed": parsed["seed"],
            "test_mse": test_mse,
            "test_mae": test_mae,
        })

        all_results = sorted(all_results, key=lambda x: (x["method"], x["seed"]))
    save_run_results_csv(all_results, os.path.join(RESULTS_DIR, "all_run_test_results.csv"))
    save_summary_csv(all_results, os.path.join(RESULTS_DIR, "all_run_summary.csv"))

    plot_metric_boxplot(
        all_results,
        metric_key="test_mse",
        ylabel="Test MSE",
        save_path=os.path.join(RESULTS_DIR, "test_mse_across_runs_boxplot.png")
    )

    plot_metric_boxplot(
        all_results,
        metric_key="test_mae",
        ylabel="Test MAE",
        save_path=os.path.join(RESULTS_DIR, "test_mae_across_runs_boxplot.png")
    )


if __name__ == "__main__":
    main()