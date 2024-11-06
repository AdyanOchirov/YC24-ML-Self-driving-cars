import os
import gzip
import polars as pl
import numpy as np
import scipy.interpolate
import torch
from train import Model, TestData, get_test_data, get_val_data, metric_df

@torch.no_grad()
def get_submission_ensemble(models: list[Model], weights: list[float], test_data: TestData):
    for model in models:
        model.eval()
    schema = {"ride_id": pl.UInt16, "stamp_ns": pl.UInt64, "x": pl.Float32, "y": pl.Float32, "yaw": pl.Float32}
    submission = pl.DataFrame(schema=schema)
    cum_sum = [0] + list(test_data.requests.group_by("ride_id", maintain_order=True).len()["len"].cum_sum())
    for i in range(len(cum_sum) - 1):
        ride_request = test_data.requests[cum_sum[i] : cum_sum[i + 1]]
        meta = test_data.meta[i : i + 1]
        locs = test_data.locs[i : i + 1]
        controls = test_data.controls[i : i + 1]
        x = np.zeros(500)
        y = np.zeros(500)
        yaw = np.zeros(500)
        for model, weight in zip(models, weights):
            x_, y_, yaw_ = model(locs, controls, meta)
            x += x_.squeeze(0).cpu().numpy() * weight
            y += y_.squeeze(0).cpu().numpy() * weight
            yaw += yaw_.squeeze(0).cpu().numpy() * weight
        x = x / sum(weights)
        y = y / sum(weights)
        yaw = yaw / sum(weights)
        xs = np.arange(0, 20_000_000_000, 40_000_000) + 40_000_000
        interpolator = scipy.interpolate.interp1d(xs, [x, y, yaw], kind="linear", axis=-1)
        guesses: np.ndarray = interpolator(ride_request["stamp_ns"])
        ride_submission = pl.DataFrame(
            {
                "ride_id": ride_request["ride_id"],
                "stamp_ns": ride_request["stamp_ns"],
                "x": guesses[0],
                "y": guesses[1],
                "yaw": guesses[2]#(guesses[2] + np.pi) % (2 * np.pi) - np.pi,
            },
            schema=schema,
        )
        submission.vstack(ride_submission, in_place=True)
    return submission

@torch.no_grad()
def save_test_submission_ensemble(models: list[Model], weights: list[float], name: str):
    sub = get_submission_ensemble(models, weights, get_test_data().to(models[0].device())).rename({"ride_id": "testcase_id"})
    sub.write_csv(os.path.join("submissions", f"{name}.csv"), float_precision=2)
    with open(os.path.join("submissions", f"{name}.csv"), "rb") as src:
        with gzip.open(os.path.join("submissions", f"{name}.csv.gz"), "wb") as dst:
            dst.writelines(src)

def ensemble():
    device = "cuda"
    model1 = Model().to(device)
    model2 = Model().to(device)
    model3 = Model().to(device)
    model4 = Model().to(device)
    model5 = Model().to(device)
    model1.load_state_dict(torch.load(os.path.join("models", "model1.pt")))
    model2.load_state_dict(torch.load(os.path.join("models", "model2.pt")))
    model3.load_state_dict(torch.load(os.path.join("models", "model3.pt")))
    model4.load_state_dict(torch.load(os.path.join("models", "model4.pt")))
    model5.load_state_dict(torch.load(os.path.join("models", "model5.pt")))
    weights = [1.0, 1.0, 1.0, 1.0, 1.0]

    val_data = get_val_data(1).to(device)

    val_submission = get_submission_ensemble([model1, model2, model3, model4, model5], weights, val_data)
    print(f"Val loss: {metric_df(val_data.requests, val_submission)['d'].mean():.3f}")
    save_test_submission_ensemble([model1, model2, model3, model4, model5], weights, "ensemble")


if __name__ == "__main__":
    ensemble()
