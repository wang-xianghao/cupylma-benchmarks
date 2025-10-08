import torch
import torchvision
import torch.nn.functional as F
import pandas as pd
import os

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from cupylma import get_available_gpus, LMA
from argparse import ArgumentParser
from time import perf_counter
from legate.timing import time


class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_stack = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=4, stride=2, padding=0),  # (8, 13, 13)
            torch.nn.ELU(),
            torch.nn.Conv2d(8, 4, kernel_size=4, stride=2, padding=0),  # (4, 5, 5)
            torch.nn.ELU(),
            torch.nn.Conv2d(4, 4, kernel_size=2, stride=1, padding=0),  # (4, 4, 4)
            torch.nn.ELU(),
            torch.nn.Conv2d(4, 4, kernel_size=2, stride=1, padding=0),  # (4, 3, 3)
            torch.nn.ELU(),
            torch.nn.Conv2d(4, 4, kernel_size=2, stride=1, padding=0),  # (4, 2, 2)
            torch.nn.ELU(),
            torch.nn.Flatten(),
            torch.nn.Linear(4 * 2 * 2, 10),  # Fully connected layer for 10 classes
        )

    def forward(self, x):
        return self.layer_stack(x)


def test(model, test_loader, device):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100.0 * correct / len(test_loader.dataset)

    return test_loss, test_acc


def train_with_lma(
    model, train_loader, test_loader, num_epochs, lr, slice_size, devices
):
    # Creat the LMA optimizer
    def residual_fn(a, b):
        return torch.sqrt(torch.nn.functional.cross_entropy(a, b, reduction="none"))

    lma = LMA(model, devices, residual_fn, learning_rate=lr)

    # Train
    timestamps, loss_values, acc_values = [], [], []
    t_all = 0
    for epoch in range(1, num_epochs + 1):
        for idx, (x, y) in enumerate(train_loader):
            t_start = time()
            loss, terminated = lma.step(x, y, slice_size=slice_size)
            t_cur = (time() - t_start) / 1e6
            t_all += t_cur

            # Sample the result per batch
            timestamps.append(t_all)
            test_loss, test_acc = test(model, test_loader, devices[0])
            loss_values.append(test_loss)
            acc_values.append(test_acc)

            # Print the result
            print(
                f"Epoch {epoch} Batch {idx}: Average loss {test_loss:10.3e}, Accuracy {test_acc}%, Time {t_cur:10.3} seconds"
            )

            if terminated:
                print("Train is early stopped.")
                return timestamps, loss_values, acc_values
        print()

    return timestamps, loss_values, acc_values


def train_with_adam(model, train_loader, test_loader, num_epochs, lr, device):
    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train
    timestamps, loss_values, acc_values = [], [], []
    t_all = 0
    for epoch in range(1, num_epochs + 1):
        t_start = perf_counter()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        t_cur = perf_counter() - t_start
        t_all += t_cur

        # Sample the result per epoch
        timestamps.append(t_all)
        test_loss, test_acc = test(model, test_loader, device)
        loss_values.append(test_loss)
        acc_values.append(test_acc)

        print(
            f"Epoch {epoch}: Average loss {test_loss:10.3e}, Accuracy {test_acc}%, Time {t_cur:10.3} seconds"
        )

    return timestamps, loss_values, acc_values


def main():
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("--optim", type=str, help="optimizer")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--slice_size", type=int, default=None, help="slice size")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument(
        "--out", type=str, default="result.csv", help="number of epochs"
    )
    args = parser.parse_args()
    optim = args.optim
    batch_size = args.batch_size
    slice_size = args.slice_size
    num_epochs = args.epochs
    lr = args.lr
    outfile = args.out

    # Prepare the dataset
    train_dataset = torchvision.datasets.MNIST(
        root=".data", train=True, transform=ToTensor(), download=False
    )
    test_dataset = torchvision.datasets.MNIST(
        root=".data", train=False, transform=ToTensor()
    )
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Build the model
    devices = get_available_gpus()
    model = SimpleCNN().to(devices[0])

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Size: {trainable_params:,}")
    print(f"Batch Size: {batch_size:,}")
    print(f"Jacobian Size: {trainable_params * batch_size * 4:,} bytes")
    print()

    # Train
    if optim == "adam":
        time_stamps, loss_values, acc_values = train_with_adam(
            model, train_loader, test_loader, num_epochs, lr, devices[0]
        )
    elif optim == "lma":
        time_stamps, loss_values, acc_values = train_with_lma(
            model, train_loader, test_loader, num_epochs, lr, slice_size, devices
        )
    else:
        raise ValueError("Undefined optimizer")

    # Write the result
    df = pd.DataFrame({"time": time_stamps, "loss": loss_values, "acc": acc_values})
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    df.to_csv(outfile, index=False)


if __name__ == "__main__":
    main()
