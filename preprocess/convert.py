import os
import json
from tqdm import tqdm
import polars as pl

def convert_train_locs():
    schema = {
        "stamp_ns": pl.UInt64,
        "x": pl.Float32,
        "y": pl.Float32,
        "z": pl.Float32,
        "roll": pl.Float32,
        "pitch": pl.Float32,
        "yaw": pl.Float32,
        "ride_id": pl.UInt16,
    }
    locs = pl.DataFrame(schema=schema)
    print("Converting train locs...")
    for i in tqdm(os.listdir(os.path.join("YaCupTrain"))):
        ride = (
            pl.read_csv(os.path.join("YaCupTrain", i, "localization.csv"))
            .with_columns(int(i))
            .rename({"literal": "ride_id"})
            .cast(schema)
        )
        locs.vstack(ride, in_place=True)
    locs = locs.sort("ride_id", "stamp_ns")[["ride_id", "stamp_ns", "x", "y", "z", "roll", "pitch", "yaw"]]
    locs.write_parquet(os.path.join("converted", "full_train_locs.parquet"))


def convert_train_controls():
    schema = {
        "stamp_ns": pl.UInt64,
        "acceleration_level": pl.Float32,
        "steering": pl.Float32,
        "ride_id": pl.UInt16,
    }
    controls = pl.DataFrame(schema=schema)
    print("Converting train controls...")   
    for i in tqdm(os.listdir(os.path.join("YaCupTrain"))):
        ride = (
            pl.read_csv(os.path.join("YaCupTrain", i, "control.csv"))
            .with_columns(int(i))
            .rename({"literal": "ride_id"})
            .cast(schema)
        )
        controls.vstack(ride, in_place=True)
    controls = controls.sort("ride_id", "stamp_ns")[["ride_id", "stamp_ns", "acceleration_level", "steering"]]
    controls.write_parquet(os.path.join("converted", "full_train_controls.parquet"))


def convert_train_meta():
    rides = {
        "ride_date": [],
        "front_tires": [],
        "rear_tires": [],
        "vehicle_id": [],
        "vehicle_model": [],
        "vehicle_model_modification": [],
        "location_reference_point_id": [],
    }
    print("Converting train meta...")
    for i in tqdm(os.listdir(os.path.join("YaCupTrain"))):
        metadata = json.load(open(os.path.join("YaCupTrain", i, "metadata.json")))
        rides["ride_date"].append(metadata["ride_date"])
        rides["front_tires"].append(metadata["tires"]["front"])
        rides["rear_tires"].append(metadata["tires"]["rear"])
        rides["vehicle_id"].append(metadata["vehicle_id"])
        rides["vehicle_model"].append(metadata["vehicle_model"])
        rides["vehicle_model_modification"].append(metadata["vehicle_model_modification"])
        rides["location_reference_point_id"].append(metadata["location_reference_point_id"])
    rides_df = pl.DataFrame(rides).with_columns(pl.col("ride_date").str.to_date())
    rides_df = rides_df.with_row_index("ride_id").cast({"ride_id": pl.UInt16})
    rides_df.write_parquet(os.path.join("meta", "full_train_rides.parquet"))


def convert_test_locs():
    schema = {
        "stamp_ns": pl.UInt64,
        "x": pl.Float32,
        "y": pl.Float32,
        "z": pl.Float32,
        "roll": pl.Float32,
        "pitch": pl.Float32,
        "yaw": pl.Float32,
        "ride_id": pl.UInt16,
    }
    locs = pl.DataFrame(schema=schema)
    print("Converting test locs...")
    for i in tqdm(os.listdir(os.path.join("YaCupTest"))):
        ride = (
            pl.read_csv(os.path.join("YaCupTest", i, "localization.csv"))
            .with_columns(int(i))
            .rename({"literal": "ride_id"})
            .cast(schema)
        )
        locs.vstack(ride, in_place=True)
    locs = locs.sort("ride_id", "stamp_ns")[["ride_id", "stamp_ns", "x", "y", "z", "roll", "pitch", "yaw"]]
    locs.write_parquet(os.path.join("converted", "test_locs.parquet"))


def convert_test_controls():
    schema = {
        "stamp_ns": pl.UInt64,
        "acceleration_level": pl.Float32,
        "steering": pl.Float32,
        "ride_id": pl.UInt16,
    }
    controls = pl.DataFrame(schema=schema)
    print("Converting test controls...")
    for i in tqdm(os.listdir(os.path.join("YaCupTest"))):
        ride = (
            pl.read_csv(os.path.join("YaCupTest", i, "control.csv"))
            .with_columns(int(i))
            .rename({"literal": "ride_id"})
            .cast(schema)
        )
        controls.vstack(ride, in_place=True)
    controls = controls.sort("ride_id", "stamp_ns")[["ride_id", "stamp_ns", "acceleration_level", "steering"]]
    controls.write_parquet(os.path.join("converted", "test_controls.parquet"))


def convert_test_meta():
    rides = {
        "ride_date": [],
        "front_tires": [],
        "rear_tires": [],
        "vehicle_id": [],
        "vehicle_model": [],
        "vehicle_model_modification": [],
        "location_reference_point_id": [],
    }
    print("Converting test meta...")
    for i in tqdm(os.listdir(os.path.join("YaCupTest"))):
        metadata = json.load(open(os.path.join("YaCupTest", i, "metadata.json")))
        rides["ride_date"].append(metadata["ride_date"])
        rides["front_tires"].append(metadata["tires"]["front"])
        rides["rear_tires"].append(metadata["tires"]["rear"])
        rides["vehicle_id"].append(metadata["vehicle_id"])
        rides["vehicle_model"].append(metadata["vehicle_model"])
        rides["vehicle_model_modification"].append(metadata["vehicle_model_modification"])
        rides["location_reference_point_id"].append(metadata["location_reference_point_id"])
    test_rides = pl.DataFrame(rides).with_columns(pl.col("ride_date").str.to_date())
    test_rides = test_rides.with_row_index("ride_id").cast({"ride_id": pl.UInt16})
    test_rides.write_parquet(os.path.join("meta", "test_rides.parquet"))


def convert_test_request():
    schema = {
        "stamp_ns": pl.UInt64,
        "ride_id": pl.UInt16,
    }
    test_requests = pl.DataFrame(schema=schema)
    print("Converting test requests...")
    for i in tqdm(os.listdir(os.path.join("YaCupTest"))):
        requests = (
            pl.read_csv(os.path.join("YaCupTest", i, "requested_stamps.csv"))
            .with_columns(int(i))
            .rename({"literal": "ride_id"})
            .cast(schema)
        )
        test_requests.vstack(requests, in_place=True)
    test_requests = test_requests.sort("ride_id", "stamp_ns")[["ride_id", "stamp_ns"]]
    test_requests.write_parquet(os.path.join("converted", "test_requests.parquet"))


def convert_all():
    os.makedirs("converted", exist_ok=True)
    os.makedirs("meta", exist_ok=True)

    convert_train_locs()
    convert_train_controls()
    convert_train_meta()
    convert_test_locs()
    convert_test_controls()
    convert_test_meta()
    convert_test_request()


if __name__ == "__main__":
    convert_all()
