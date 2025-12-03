import pandas as pd
import matplotlib.pyplot as plt
import argparse
import scienceplots

plt.style.use("science")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Plot training loss and accuracy from CSV files"
    )
    parser.add_argument(
        "--file", type=str, nargs="+", required=True, help="Input CSV file paths"
    )
    parser.add_argument(
        "--label", type=str, nargs="+", required=True, help="Labels for each file"
    )
    args = parser.parse_args()

    # Check that number of files matches number of labels
    if len(args.file) != len(args.label):
        raise ValueError(
            f"Number of files ({len(args.file)}) must match number of labels ({len(args.label)})"
        )

    # Create first plot: time vs loss
    plt.figure()
    for file_path, label in zip(args.file, args.label):
        df = pd.read_csv(file_path)
        plt.plot(df["time"], df["loss"], label=label)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Loss")
    plt.title("Loss over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss.svg")
    plt.show()

    # Create second plot: time vs accuracy
    plt.figure()
    for file_path, label in zip(args.file, args.label):
        df = pd.read_csv(file_path)
        plt.plot(df["time"], df["acc"], label=label)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy.svg")
    plt.show()


if __name__ == "__main__":
    main()
