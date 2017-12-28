from data_loader import DataLoader
from resnet import ResNet
import numpy as np


def run_resnet():
    dl = DataLoader("Selfie-dataset")
    rs = ResNet()
    dl.read_data()
    dl.create_train_test()
    dl.shuffle()
    dl.create_train_test_df()
    dl.create_train_test_data()
    predicted_values = []
    for i in range(1, 101):
        predicted_values.append(rs.resNet_50(dl.get_data(100)))
    print("Predicted Values ", predicted_values)
    np.savetxt("results.txt", np.asarray(predicted_values))




if __name__ == "__main__":
    run_resnet()
