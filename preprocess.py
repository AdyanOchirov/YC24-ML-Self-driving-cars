from preprocess.convert import convert_all
from preprocess.val_split import split
from preprocess.interpolate import interpolate

if __name__ == "__main__":
    convert_all()
    print("Doing train/val splitting...")
    split()
    print("Interpolating...")
    interpolate()
