import os
import gzip
import polars as pl
import numpy as np
from tqdm import tqdm
import scipy.interpolate
import torch

def speed(x: torch.Tensor) -> torch.Tensor:
    v_start = (-3 / 2) * x[..., 0] + 2 * x[..., 1] - (1 / 2) * x[..., 2]
    v_mid = (x[..., 2:] - x[..., :-2]) / 2
    v_end = (3 / 2) * x[..., -1] - 2 * x[..., -2] + (1 / 2) * x[..., -3]
    v = torch.cat([v_start.unsqueeze(-1), v_mid, v_end.unsqueeze(-1)], -1)
    return v

def rotate(locs: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    x, y, rest, yaw = locs[..., 0:1, :], locs[..., 1:2, :], locs[..., 2:-1, :], locs[..., -1:, :]
    x, y = x * torch.cos(angle) - y * torch.sin(angle), x * torch.sin(angle) + y * torch.cos(angle)
    yaw = yaw + angle
    return torch.cat([x, y, rest, yaw], dim=-2)

def loss_fn(
    locs: torch.Tensor,
    pred_x: torch.Tensor,
    pred_y: torch.Tensor,
    pred_yaw: torch.Tensor,
) -> torch.Tensor:
    x, y, yaw = locs[..., 0, :], locs[..., 1, :], locs[..., 5, :]
    x, y, yaw, pred_x, pred_y, pred_yaw = (
        x[..., -375:],
        y[..., -375:],
        yaw[..., -375:],
        pred_x[..., -375:],
        pred_y[..., -375:],
        pred_yaw[..., -375:],
    )
    d1 = (x - pred_x) ** 2 + (y - pred_y) ** 2
    d2 = (x + torch.cos(yaw) - pred_x - torch.cos(pred_yaw)) ** 2 + (
        y + torch.sin(yaw) - pred_y - torch.sin(pred_yaw)
    ) ** 2
    return torch.sqrt((d1 + d2) / 2 + 1e-6).mean(-1)

def preprocess(locs: torch.Tensor, controls: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x, y, z, roll, pitch, yaw = locs.split(1, -2)
    acc, steer = controls.split(1, -2)
    acc_pos = acc * (acc > 0)
    acc_neg = acc * (acc < 0)

    v_x, v_y, v_yaw = speed(x), speed(y), speed(yaw)
    abs_v = torch.sqrt(v_x**2 + v_y**2 + 1e-6)
    v_abs_v = speed(abs_v)

    acc_pos = acc_pos / 30000
    acc_neg = acc_neg / 6000
    steer = steer / 360

    locs = torch.cat([x, y, z, roll, pitch, yaw, acc_pos[..., :125], acc_neg[..., :125], steer[..., :125], v_x, v_y, abs_v, v_abs_v, v_yaw], -2)
    controls = torch.cat([acc_pos, acc_neg, steer], -2)

    return locs, controls


def preprocess_train(locs: torch.Tensor, controls: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inp, controls = preprocess(locs[..., :125], controls)

    return inp, locs, controls


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, locs: pl.DataFrame, controls: pl.DataFrame, meta: pl.DataFrame):
        self.meta = torch.tensor(
            meta.with_columns(pl.col("ride_date").dt.month().alias("month"))[
                ["front_tires", "rear_tires", "vehicle_model_modification", "month"]
            ].to_numpy()
        )
        locs_np = locs.drop("ride_id").to_numpy().reshape(-1, 1500, 6).transpose(0, 2, 1)
        locs_np[..., -1, :] = np.unwrap(locs_np[..., -1, :])
        self.locs = torch.tensor(locs_np)
        self.controls = torch.tensor(controls.drop("ride_id").to_numpy().reshape(-1, 1500, 2).transpose(0, 2, 1))

    def __len__(self):
        return self.locs.shape[0]

    def __getitem__(self, idx):
        return self.locs[idx], self.controls[idx], self.meta[idx]


def get_train_data() -> TrainDataset:
    meta = pl.read_parquet(os.path.join("meta", "train_rides.parquet"))
    locs = pl.read_parquet(os.path.join("interpolated", "train_locs.parquet"))
    controls = pl.read_parquet(os.path.join("interpolated", "train_controls.parquet"))
    return TrainDataset(locs, controls, meta)

class TestData:
    def __init__(self, meta: pl.DataFrame, locs: pl.DataFrame, controls: pl.DataFrame, requests: pl.DataFrame):
        self.meta = torch.tensor(
            meta.with_columns(pl.col("ride_date").dt.month().alias("month"))[
                ["front_tires", "rear_tires", "vehicle_model_modification", "month"]
            ].to_numpy()
        )
        locs_np = locs.drop("ride_id").to_numpy().reshape(-1, 125, 6).transpose(0, 2, 1)
        locs_np[..., -1, :] = np.unwrap(locs_np[..., -1, :])
        self.controls = torch.tensor(controls.drop("ride_id").to_numpy().reshape(-1, 500, 2).transpose(0, 2, 1))
        self.requests = requests

        self.locs, self.controls = preprocess(torch.tensor(locs_np), self.controls)

    def to(self, device):
        self.meta = self.meta.to(device)
        self.locs = self.locs.to(device)
        self.controls = self.controls.to(device)
        return self


def get_val_data(i) -> TestData:
    val_rides = pl.read_parquet(os.path.join("meta", "val_rides.parquet"))
    val_locs = pl.read_parquet(os.path.join("interpolated", f"val_locs_{i}.parquet"))
    val_controls = pl.read_parquet(os.path.join("interpolated", f"val_controls_{i}.parquet"))
    val_requests = pl.read_parquet(os.path.join("data", f"val_requests_{i}.parquet"))
    data = TestData(val_rides, val_locs, val_controls, val_requests)
    return data


def get_test_data() -> TestData:
    test_rides = pl.read_parquet(os.path.join("meta", "test_rides.parquet"))
    test_locs = pl.read_parquet(os.path.join("interpolated", "test_locs.parquet"))
    test_controls = pl.read_parquet(os.path.join("interpolated", "test_controls.parquet"))
    test_requests = pl.read_parquet(os.path.join("converted", "test_requests.parquet"))
    data = TestData(test_rides, test_locs, test_controls, test_requests)
    return data


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Sequential(
            torch.nn.Conv1d(14 + 16 + 12, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(64, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(64, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(64, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(64, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(64, 16, 3, padding=1),
        )
        self.layers = torch.nn.Sequential(
            torch.nn.Conv1d(16 + 3 + 16 + 12, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 2, 3, padding=1),
        )
        self.c = torch.nn.Parameter(torch.tensor([[[0.1], [0.1], [1], [50], [50], [1], [1], [1], [1], [10], [10], [10], [1000], [100]]]), requires_grad=False)

        self.pos_emb1 = torch.nn.Parameter(torch.randn(1, 16, 125))
        self.pos_emb2 = torch.nn.Parameter(torch.randn(1, 16, 500))
        self.tire_emb = torch.nn.Embedding(14, 4)
        self.modif_emb = torch.nn.Embedding(6, 4)

    def device(self):
        return next(self.parameters()).device

    def forward(self, locs: torch.Tensor, controls: torch.Tensor, meta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xyz_mean = locs[..., :3, :].mean(-1, keepdim=True)
        inp = torch.cat([locs[..., :3, :] - xyz_mean, locs[..., 3:, :]], -2)
        inp = inp * self.c

        front_tire = self.tire_emb(meta[..., 0] % 14)
        rear_tire = self.tire_emb(meta[..., 1] % 14)
        modif = self.modif_emb(meta[..., 2])
        meta = torch.cat([front_tire, rear_tire, modif], -1).unsqueeze(-1)

        pos_emb_1 = self.pos_emb1.repeat(inp.shape[0], 1, 1)
        meta_1 = meta.repeat(1, 1, 125)
        inp = torch.cat([inp, pos_emb_1, meta_1], -2)
        inp = self.emb(inp)

        inp = inp.mean(-1, keepdim=True)
        inp = inp.repeat(1, 1, 500)
        pos_emb_2 = self.pos_emb2.repeat(inp.shape[0], 1, 1)
        meta_2 = meta.repeat(1, 1, 500)
        inp = torch.cat([inp, controls, pos_emb_2, meta_2], -2)
        inp = self.layers(inp)

        pred_v_abs_v, pred_v_yaw = inp.split(1, 1)
        pred_v_abs_v = torch.nn.Tanh()(pred_v_abs_v) * 0.0095 - 0.003
        pred_v_yaw = torch.nn.Tanh()(pred_v_yaw) * 0.032

        pred_abs_v = locs[..., -3:-2,-1:] + pred_v_abs_v[..., 125:].cumsum(-1)
        if not self.training:
            pred_abs_v = torch.clip(pred_abs_v, 0, 1.05)

        pred_yaw = locs[..., 5:6, -1:] + pred_v_yaw[..., 125:].cumsum(-1)
        pred_v_x = pred_abs_v * torch.cos(pred_yaw)
        pred_v_y = pred_abs_v * torch.sin(pred_yaw)
        pred_x = locs[..., 0:1, -1:] + pred_v_x.cumsum(-1)
        pred_y = locs[..., 1:2, -1:] + pred_v_y.cumsum(-1)

        x = torch.cat([locs[..., 0:1, :], pred_x], -1)
        y = torch.cat([locs[..., 1:2, :], pred_y], -1)
        yaw = torch.cat([locs[..., 5:6, :], pred_yaw], -1)

        return x.squeeze(1), y.squeeze(1), yaw.squeeze(1)


@torch.no_grad()
def get_submission(model: Model, test_data: TestData) -> pl.DataFrame:
    model.eval()
    schema = {"ride_id": pl.UInt16, "stamp_ns": pl.UInt64, "x": pl.Float32, "y": pl.Float32, "yaw": pl.Float32}
    submission = pl.DataFrame(schema=schema)
    cum_sum = [0] + list(test_data.requests.group_by("ride_id", maintain_order=True).len()["len"].cum_sum())
    for i in range(len(cum_sum) - 1):
        ride_request = test_data.requests[cum_sum[i] : cum_sum[i + 1]]
        meta = test_data.meta[i : i + 1]
        locs = test_data.locs[i : i + 1]
        controls = test_data.controls[i : i + 1]
        x, y, yaw = model(locs, controls, meta)
        x, y, yaw = x.squeeze(0).cpu().numpy(), y.squeeze(0).cpu().numpy(), yaw.squeeze(0).cpu().numpy()
        xs = np.arange(0, 20_000_000_000, 40_000_000) + 40_000_000
        interpolator = scipy.interpolate.interp1d(xs, [x, y, yaw], kind="linear", axis=-1)
        guesses: np.ndarray = interpolator(ride_request["stamp_ns"])
        ride_submission = pl.DataFrame(
            {
                "ride_id": ride_request["ride_id"],
                "stamp_ns": ride_request["stamp_ns"],
                "x": guesses[0],
                "y": guesses[1],
                "yaw": (guesses[2] + np.pi) % (2 * np.pi) - np.pi,
            },
            schema=schema,
        )
        submission.vstack(ride_submission, in_place=True)
    return submission

def metric_df(request: pl.DataFrame, submission: pl.DataFrame) -> pl.DataFrame:
    d1 = (request["x"] - submission["x"]) ** 2 + (request["y"] - submission["y"]) ** 2
    d2 = (request["x"] + request["yaw"].cos() - submission["x"] - submission["yaw"].cos()) ** 2 + (
        request["y"] + request["yaw"].sin() - submission["y"] - submission["yaw"].sin()
    ) ** 2
    return pl.DataFrame({"ride_id": request["ride_id"], "d": ((d1 + d2) / 2).sqrt()}).group_by("ride_id").agg(pl.col("d").mean())

@torch.no_grad()
def val_loss(model: Model, data: TestData):
    sub = get_submission(model, data)
    return metric_df(data.requests, sub)["d"].mean()

@torch.no_grad()
def save_test_submission(model: Model, name: str):
    sub = get_submission(model, get_test_data().to(model.device())).rename({"ride_id": "testcase_id"})
    sub.write_csv(os.path.join("submissions", f"{name}.csv"), float_precision=2)
    with open(os.path.join("submissions", f"{name}.csv"), "rb") as src:
        with gzip.open(os.path.join("submissions", f"{name}.csv.gz"), "wb") as dst:
            dst.writelines(src)

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(name, epochs, batch_size, lr, pct_start, seed, val_freq,  device="cuda"):
    set_seed(seed)

    model = Model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    data = get_train_data()
    val_data = get_val_data(1).to(device)
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, epochs=epochs, steps_per_epoch=len(loader), pct_start=pct_start)

    model.train()
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        for _, (locs, controls, meta) in enumerate(loader):
            i = torch.randint(0, 1000, (1,)).item()
            locs, controls = locs[..., i : i + 500], controls[..., i : i + 500]
            angle = torch.rand(1) * torch.pi * 2
            locs = rotate(locs, angle)
            inp, locs, controls = preprocess_train(locs, controls)
            inp, locs, controls, meta = inp.to(device), locs.to(device), controls.to(device), meta.to(device)
            pred = model(inp, controls, meta)
            loss = loss_fn(locs, *pred).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if (epoch + 1) % val_freq == 0:
            pbar.set_description(f"Val loss@{epoch + 1}: {val_loss(model, val_data):.3f}")
        pbar.update()

    os.makedirs("models", exist_ok=True)
    os.makedirs("submissions", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("models", f"{name}.pt"))
    save_test_submission(model, name)


if __name__ == "__main__":
    train("model1", 2000, 128, 3e-4, 0.1, 451, 50)
    train("model2", 1500, 64, 3e-4, 0.1, 452, 50)
    train("model3", 2000, 128, 3e-4, 0.3, 452, 50)
    train("model4", 3000, 128, 3e-4, 0.25, 454, 100)
    train("model5", 3000, 128, 3e-4, 0.25, 453, 100)
