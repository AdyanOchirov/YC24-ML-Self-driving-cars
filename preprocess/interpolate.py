import os
import polars as pl
import numpy as np
import scipy.interpolate

def interpolate_controls_test(controls: pl.DataFrame) -> pl.DataFrame:
    schema = controls.drop("stamp_ns").schema
    xs = np.arange(0, 20_000_000_000, 40_000_000) + 40_000_000
    cum_sum = [0] + list(controls.group_by("ride_id", maintain_order=True).len()["len"].cum_sum())
    interp_df = pl.DataFrame(schema=schema)
    for i in range(len(cum_sum) - 1):
        ride_df = controls[cum_sum[i] : cum_sum[i + 1]]
        interpolator = scipy.interpolate.interp1d(
            ride_df["stamp_ns"],
            ride_df[["acceleration_level", "steering"]],
            kind="linear",
            axis=0,
            fill_value="extrapolate",
            assume_sorted=True,
        )
        interpolation = interpolator(xs)
        interp_df.vstack(
            pl.DataFrame(
                {
                    "ride_id": [ride_df["ride_id"][0]] * 500,
                    "acceleration_level": interpolation[:, 0],
                    "steering": interpolation[:, 1],
                }
            ).cast(schema),
            in_place=True,
        )
    return interp_df


def interpolate_locs_test(locs: pl.DataFrame) -> pl.DataFrame:
    schema = locs.drop("stamp_ns").schema
    xs = np.arange(0, 5_000_000_000, 40_000_000) + 40_000_000
    cum_sum = [0] + list(locs.group_by("ride_id", maintain_order=True).len()["len"].cum_sum())
    interp_df = pl.DataFrame(schema=schema)
    for i in range(len(cum_sum) - 1):
        ride_df = locs[cum_sum[i] : cum_sum[i + 1]].with_columns(
            pl.col("yaw").sin().alias("yaw_sin"),
            pl.col("yaw").cos().alias("yaw_cos"),
        )
        interpolator = scipy.interpolate.interp1d(
            ride_df["stamp_ns"],
            ride_df[["x", "y", "z", "roll", "pitch", "yaw_sin", "yaw_cos"]],
            kind="linear",
            axis=0,
            fill_value="extrapolate",
            assume_sorted=True,
        )
        interpolation = interpolator(xs)
        interp_df.vstack(
            pl.DataFrame(
                {
                    "ride_id": [ride_df["ride_id"][0]] * 125,
                    "x": interpolation[:, 0],
                    "y": interpolation[:, 1],
                    "z": interpolation[:, 2],
                    "roll": interpolation[:, 3],
                    "pitch": interpolation[:, 4],
                    "yaw": np.arctan2(interpolation[:, 5], interpolation[:, 6]),
                }
            ).cast(schema),
            in_place=True,
        )
    return interp_df


def interpolate_controls_train(controls: pl.DataFrame) -> pl.DataFrame:
    schema = controls.drop("stamp_ns").schema
    cum_sum = [0] + list(controls.group_by("ride_id", maintain_order=True).len()["len"].cum_sum())
    interp_df = pl.DataFrame(schema=schema)
    for i in range(len(cum_sum) - 1):
        ride_df = controls[cum_sum[i] : cum_sum[i + 1]]
        interpolator = scipy.interpolate.interp1d(
            ride_df["stamp_ns"],
            ride_df[["acceleration_level", "steering"]],
            kind="linear",
            axis=0,
            fill_value="extrapolate",
            assume_sorted=True,
        )
        xs = np.arange(0, 60_000_000_000, 40_000_000) + 40_000_000
        interpolation = interpolator(xs)
        interp_df.vstack(
            pl.DataFrame(
                {
                    "ride_id": [ride_df["ride_id"][0]] * 1500,
                    "acceleration_level": interpolation[:, 0],
                    "steering": interpolation[:, 1],
                }
            ).cast(schema),
            in_place=True,
        )
    return interp_df


def interpolate_locs_train(locs: pl.DataFrame) -> pl.DataFrame:
    schema = locs.drop("stamp_ns").schema
    cum_sum = [0] + list(locs.group_by("ride_id", maintain_order=True).len()["len"].cum_sum())
    interp_df = pl.DataFrame(schema=schema)
    for i in range(len(cum_sum) - 1):
        ride_df = locs[cum_sum[i] : cum_sum[i + 1]].with_columns(
            pl.col("yaw").sin().alias("yaw_sin"),
            pl.col("yaw").cos().alias("yaw_cos"),
        )
        interpolator = scipy.interpolate.interp1d(
            ride_df["stamp_ns"],
            ride_df[["x", "y", "z", "roll", "pitch", "yaw_sin", "yaw_cos"]],
            kind="linear",
            axis=0,
            fill_value="extrapolate",
            assume_sorted=True,
        )
        xs = np.arange(0, 60_000_000_000, 40_000_000) + 40_000_000
        interpolation = interpolator(xs)
        interp_df.vstack(
            pl.DataFrame(
                {
                    "ride_id": [ride_df["ride_id"][0]] * 1500,
                    "x": interpolation[:, 0],
                    "y": interpolation[:, 1],
                    "z": interpolation[:, 2],
                    "roll": interpolation[:, 3],
                    "pitch": interpolation[:, 4],
                    "yaw": np.arctan2(interpolation[:, 5], interpolation[:, 6]),
                }
            ).cast(schema),
            in_place=True,
        )
    return interp_df


def interpolate():
    os.makedirs("interpolated", exist_ok=True)

    interpolate_controls_test(pl.read_parquet(os.path.join("data", "val_controls_0.parquet"))).write_parquet(os.path.join("interpolated", "val_controls_0.parquet"))
    interpolate_controls_test(pl.read_parquet(os.path.join("data", "val_controls_1.parquet"))).write_parquet(os.path.join("interpolated", "val_controls_1.parquet"))
    interpolate_controls_test(pl.read_parquet(os.path.join("data", "val_controls_2.parquet"))).write_parquet(os.path.join("interpolated", "val_controls_2.parquet"))
    interpolate_controls_test(pl.read_parquet(os.path.join("converted", "test_controls.parquet"))).write_parquet(os.path.join("interpolated", "test_controls.parquet"))

    interpolate_locs_test(pl.read_parquet(os.path.join("data", "val_locs_0.parquet"))).write_parquet(os.path.join("interpolated", "val_locs_0.parquet"))
    interpolate_locs_test(pl.read_parquet(os.path.join("data", "val_locs_1.parquet"))).write_parquet(os.path.join("interpolated", "val_locs_1.parquet"))
    interpolate_locs_test(pl.read_parquet(os.path.join("data", "val_locs_2.parquet"))).write_parquet(os.path.join("interpolated", "val_locs_2.parquet"))
    interpolate_locs_test(pl.read_parquet(os.path.join("converted", "test_locs.parquet"))).write_parquet(os.path.join("interpolated", "test_locs.parquet"))

    interpolate_controls_train(pl.read_parquet(os.path.join("data", "train_controls.parquet"))).write_parquet(os.path.join("interpolated", "train_controls.parquet"))
    interpolate_locs_train(pl.read_parquet(os.path.join("data", "train_locs.parquet"))).write_parquet(os.path.join("interpolated", "train_locs.parquet"))


if __name__ == "__main__":
    interpolate()
