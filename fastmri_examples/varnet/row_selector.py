import torch
import h5py
import pathlib
from fastmri.data.mri_data import fetch_dir
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

def train():

    path_config = pathlib.Path("../../fastmri_dirs.yaml")
    data_path = fetch_dir("knee_path", path_config)
    training_path = data_path / "singlecoil_train"
    validation_path = data_path / "singlecoil_val"

    model = nn.Sequential(nn.Linear(640, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 640), nn.Softmax())
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(2):
        print("Epoch:", epoch)
        file_num = 1
        for path in training_path.iterdir():
            print("\nFile Number:", file_num)
            file_num += 1
            file_loss = 0
            file = h5py.File(path, mode="r")
            kspace = file['kspace']
            for i in range(5, kspace.shape[0] - 5):
                slice = torch.from_numpy(np.array(kspace[i]))
                row_sums = torch.sum(torch.sqrt(slice.real ** 2 + slice.imag ** 2), 1)
                visible = torch.zeros(row_sums.shape[0])
                start, end = 320-50, 320+50
                visible[start:end] = row_sums[start:end]
                row_sums[start:end] = 0
                for run in range(62):
                    optimizer.zero_grad()
                    output = model(visible)
                    loss = pickingLoss(output, row_sums)
                    file_loss += loss
                    loss.backward()
                    optimizer.step()
                    pick = torch.argmax(output)
                    if row_sums[pick] != 0:
                        visible[pick] = row_sums[pick]
                        row_sums[pick] = 0
            print("\n\nAverage Loss:", (file_loss/(kspace.shape[0] - 10)))
    torch.save(model.state_dict(), "state_dict_model.pt")

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