import torch
import h5py
import pathlib
from fastmri.data.mri_data import fetch_dir
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import random

def train():

    path_config = pathlib.Path("../../fastmri_dirs.yaml")
    data_path = fetch_dir("knee_path", path_config)
    training_path = data_path / "singlecoil_train"
    validation_path = data_path / "singlecoil_val"
    softmax = nn.Softmax(dim=None)
    criterion = nn.KLDivLoss(reduction="batchmean")
    model = nn.Sequential(nn.Linear(640, 2048), nn.ReLU(), nn.Linear(2048, 2048), nn.ReLU(), nn.Linear(2048, 640), nn.LogSoftmax(dim=None))
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(30):
        print("Epoch:", epoch)
        file_loss = 0
        for path in training_path.iterdir():
            file = h5py.File(path, mode="r")
            kspace = file['kspace']
            index = random.randint(5, kspace.shape[0] - 5)

            slice = torch.from_numpy(np.array(kspace[index]))
            row_sums = torch.sum(torch.sqrt(slice.real ** 2 + slice.imag ** 2), 1)
            visible = torch.zeros(row_sums.shape[0])
            start, end = 320-50, 320+50
            visible[start:end] = row_sums[start:end]
            row_sums[start:end] = 0
            indexes = torch.topk(row_sums, 62)[1].numpy()
            for run in range(62):
                if random.randint(0, 3) == 1:
                    visible[indexes[run]] = row_sums[indexes[run]]
                    row_sums[indexes[run]] = 0

            optimizer.zero_grad()
            output = model(visible)
            loss = criterion(output, softmax(row_sums))
            file_loss += loss
            loss.backward()
            optimizer.step()
        print("\n\nAverage Loss:", (file_loss/973))
    torch.save(model.state_dict(), "state_dict_model2.pt")

    # for key, val in counter.items():
    #     if val > 22500:
    #         print("(" + str(key) + ", " + str(val) + "), ", sep="")
    # plt.title("Top 162 Row Sum vs Row Index")
    # plt.xlabel("Row Index")
    # plt.ylabel("Times in top 162")
    # plt.plot(list(range(640)), counter)
    # plt.savefig("Top162x.png")
    # plt.close()


def pickingLoss(output, target):
    scaling_factor = 0.1
    maximum = torch.max(target)
    return (10.0*maximum - torch.sum(output*target*10.0))/maximum




if __name__ == "__main__":
    train()