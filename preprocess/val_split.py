import os
import polars as pl

NS_PER_S = 1_000_000_000

def new_stamp(i):
    return pl.col("stamp_ns") - (i * 20) * NS_PER_S

def locs_subpart(locs: pl.DataFrame, i) -> pl.DataFrame:
    return locs.filter(
        (pl.col("stamp_ns") < (i * 20 + 5) * NS_PER_S) & (pl.col("stamp_ns") >= (i * 20) * NS_PER_S)
    ).with_columns(new_stamp(i))


def controls_subpart(controls: pl.DataFrame, i) -> pl.DataFrame:
    return controls.filter(
        (pl.col("stamp_ns") < (i * 20 + 20) * NS_PER_S) & (pl.col("stamp_ns") >= (i * 20) * NS_PER_S)
    ).with_columns(new_stamp(i))


def requests_subpart(locs: pl.DataFrame, i) -> pl.DataFrame:
    return locs.filter(
        (pl.col("stamp_ns") < (i * 20 + 20) * NS_PER_S) & (pl.col("stamp_ns") >= (i * 20 + 5) * NS_PER_S)
    ).with_columns(new_stamp(i))[["ride_id", "stamp_ns", "x", "y", "yaw"]]


def split():
    os.makedirs("data", exist_ok=True)

    full_train_rides = pl.read_parquet(os.path.join("meta", "full_train_rides.parquet"))

    full_train_controls = pl.read_parquet(os.path.join("converted", "full_train_controls.parquet"))
    full_train_locs = pl.read_parquet(os.path.join("converted", "full_train_locs.parquet"))

    first_control_stamp = full_train_controls.group_by("ride_id").agg(pl.col("stamp_ns").min().alias("first_control_stamp"))
    full_train_controls = (
        full_train_controls.join(first_control_stamp, on="ride_id", how="left")
        .filter(pl.col("stamp_ns") > pl.col("first_control_stamp"))
        .with_columns(pl.col("stamp_ns") - pl.col("first_control_stamp").alias("stamp_ns"))
        .drop("first_control_stamp")
    )
    full_train_locs = (
        full_train_locs.join(first_control_stamp, on="ride_id", how="left")
        .filter(pl.col("stamp_ns") > pl.col("first_control_stamp"))
        .with_columns(pl.col("stamp_ns") - pl.col("first_control_stamp").alias("stamp_ns"))
        .drop("first_control_stamp")
    )

    val_rides = full_train_rides.sample(2000, seed=451)
    train_rides = full_train_rides.filter(~pl.col("ride_id").is_in(val_rides["ride_id"]))

    val_controls = full_train_controls.filter(pl.col("ride_id").is_in(val_rides["ride_id"]))
    train_controls = full_train_controls.filter(~pl.col("ride_id").is_in(val_controls["ride_id"]))

    val_locs = full_train_locs.filter(pl.col("ride_id").is_in(val_rides["ride_id"]))
    train_locs = full_train_locs.filter(~pl.col("ride_id").is_in(val_locs["ride_id"]))

    val_rides.write_parquet(os.path.join("meta", "val_rides.parquet"))

    for i in range(3):
        controls_subpart(val_controls, i).write_parquet(os.path.join("data", f"val_controls_{i}.parquet"))
        locs_subpart(val_locs, i).write_parquet(os.path.join("data", f"val_locs_{i}.parquet"))
        requests_subpart(val_locs, i).write_parquet(os.path.join("data", f"val_requests_{i}.parquet"))

    train_rides.write_parquet(os.path.join("meta", "train_rides.parquet"))

    train_controls.write_parquet(os.path.join("data", "train_controls.parquet"))
    train_locs.write_parquet(os.path.join("data", "train_locs.parquet"))


if __name__ == "__main__":
    split()
