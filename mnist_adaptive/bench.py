import torch
import torchvision
import torch.nn.functional as F
import pandas as pd
import os

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from cupylma import get_available_gpus, LMA
from argparse import ArgumentParser
from legate.timing import time
from legate.core import get_machine, TaskTarget

torch.manual_seed(0)
torch.cuda.manual_seed(0)

# class SimpleDNN(torch.nn.Module):
#     def __init__(self):
#         super(MNISTNet, self).__init__()
#         self.fc1 = torch.nn.Linear(784, 1)  # 784*1 + 1 = 785 parameters
#         self.relu = torch.nn.ReLU()
#         self.fc2 = torch.nn.Linear(1, 10)   # 1*10 + 10 = 20 parameters

#     def forward(self, x):
#         x = x.view(-1, 784)  # Flatten
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x


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


def get_train_loader(train_dataset, batch_size):
    return DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )


def test(model, test_loader, device):
    model.eval()
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
    model,
    train_dataset,
    test_loader,
    batch_start,
    batch_end,
    batch_slope,
    slice_size,
    num_epochs,
    learning_rate,
    devices,
):
    # Creat the LMA optimizer
    def residual_fn(a, b):
        return torch.sqrt(torch.nn.functional.cross_entropy(a, b, reduction="none"))

    lma = LMA(model, devices, residual_fn, learning_rate=learning_rate)

    # Train
    decreases = 0
    batch_size = batch_start
    train_loader = get_train_loader(train_dataset, batch_size)
    timestamps, loss_values, acc_values, batches = [], [], [], []
    t_all = 0
    for epoch in range(1, num_epochs + 1):
        for idx, (x, y) in enumerate(train_loader):
            model.train()
            t_start = time()
            loss, terminated = lma.step(x, y, slice_size=slice_size)
            t_cur = (time() - t_start) / 1e6
            t_all += t_cur

            # Sample the result per batch
            timestamps.append(t_all)
            test_loss, test_acc = test(model, test_loader, devices[0])
            loss_values.append(test_loss)
            acc_values.append(test_acc)
            batches.append(batch_size)

            if len(loss_values) > 1 and loss_values[-1] > loss_values[-2]:
                decreases += 1

            # Print the result
            print(
                f"Epoch {epoch} Batch {idx} Batch Size {batch_size}: Average loss {test_loss:10.3e}, Accuracy {test_acc:5.2f}%, Time {t_cur:6.3} seconds"
            )

            if terminated or decreases > 3:
                decreases = 0
                batch_size = int(batch_size * batch_slope)
                if batch_size > batch_end:
                    print("Batch size reaches the limit! Train terminated.")
                    return timestamps, loss_values, acc_values, batches
                print(f"Batch size increases to {batch_size:,}")
                train_loader = get_train_loader(train_dataset, batch_size)
                break
        print()

    return timestamps, loss_values, acc_values, batches


def main():
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--batch-start",
        type=int,
        default=200,
        required=True,
        help="beginning batch size",
    )
    parser.add_argument(
        "--batch-end", type=int, default=60000, required=True, help="ending batch size"
    )
    parser.add_argument(
        "--batch-slope",
        type=float,
        default=1.5,
        required=True,
        help="batch size increasing slope",
    )
    parser.add_argument("--slice-size", type=int, default=1024, help="slice size")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument(
        "--learning-rate", type=float, default=1.0, help="learning rate"
    )
    parser.add_argument("-o", type=str, required=True, help="file to store results")

    args = parser.parse_args()
    batch_start = args.batch_start
    batch_end = args.batch_end
    batch_slope = args.batch_slope
    slice_size = args.slice_size
    num_epochs = args.epochs
    learning_rate = args.learning_rate
    outfile = args.o

    # Prepare the dataset
    train_dataset = torchvision.datasets.MNIST(
        root=".data", train=True, transform=ToTensor(), download=False
    )
    test_dataset = torchvision.datasets.MNIST(
        root=".data", train=False, transform=ToTensor()
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=1024, shuffle=False)

    # Build the model
    devices = get_available_gpus()
    model = SimpleCNN().to(devices[0])

    print(f"Model Component (PyTorch) GPUs: {len(devices)}")
    print(f"Model Component Master Device: {devices[0]}")
    print(
        f"Optimizer Component (cuPyNumeric) GPUs: {get_machine().count(TaskTarget.GPU)}"
    )
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Size: {trainable_params:,}")
    print(f"Batch Size: {batch_start:,}...{batch_end:,}")
    print(
        f"Jacobian Size: {trainable_params * batch_start * 4:,}...{trainable_params * batch_end * 4:,} bytes"
    )
    print()

    # train
    timestamps, loss_values, acc_values, batches = train_with_lma(
        model,
        train_dataset,
        test_loader,
        batch_start,
        batch_end,
        batch_slope,
        slice_size,
        num_epochs,
        learning_rate,
        devices,
    )

    # Write the result
    df = pd.DataFrame(
        {"time": timestamps, "loss": loss_values, "acc": acc_values, "batch": batches}
    )
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    df.to_csv(outfile, index=False)


if __name__ == "__main__":
    main()
