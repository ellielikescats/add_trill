import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"


class CustomMidiDataset(Dataset):
    def __init__(self):
        pkl_path = 'preprocess_data/data.pkl'
        if not os.path.exists(pkl_path):
            print("=============================================")
            print("No pkl file to load data from!!! Stopping!")
            print("=============================================")
            raise Exception
        else:
            with open(pkl_path, "rb") as input_file:
                pickle_list = pickle.load(input_file)

        # Look at the balance between positive and negative examples:
        # Create a sublist for positive and negative examples
        positives = []
        negatives = []
        # Iterate over pickle_list that contains the 80.00 only velocities and the target 80.00 + 127.00 velocities
        for element in pickle_list:
            x = element[0]
            y = element[1]
            # Where Y and X are equal, the examples are negative
            if np.array_equal(x, y):
                # NEGATIVE
                negatives.append(element)
            # Where Y and X are not equal, the examples are positive
            else:
                # POSITIVE
                positives.append(element)

        self.X = []
        self.Y = []
        # Append a smaller amount of positives X and Y, to balance out the positive and negative examples in X and Y
        num_positives = len(positives)
        for i in range(num_positives):
            self.X.append(positives[i][0])
            self.Y.append(positives[i][1])

            self.X.append(negatives[i][0])
            self.Y.append(negatives[i][1])

        # If positive and negative examples are equally balanced in the data anyway, comment out above^ and use this:
        # for element in pickle_list:
        #     self.X.append(element[0])
        #     self.Y.append(element[1])

        # Convert to double
        self.X = np.array(self.X, dtype=np.float32)
        self.Y = np.array(self.Y, dtype=np.float32)

        self.max_values_X = []
        self.max_values_Y = []

        # Map to [0...1] column-wise to normalise
        for column in range(self.X.shape[2]):
            maxX = np.max(self.X[:, :, column])
            self.max_values_X.append(maxX)
            self.X[:, :, column] /= maxX

            maxY = np.max(self.Y[:, :, column])
            self.max_values_Y.append(maxY)
            self.Y[:, :, column] /= maxY

        # Convert arrays into tensors
        self.X = torch.tensor(self.X).to(device)
        self.Y = torch.tensor(self.Y).to(device)

    # Number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # Get a row at an index
    def __getitem__(self, idx):
        input = self.X[idx]
        label = self.Y[idx]
        sample = {"input": input, "label": label}
        return sample
