import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from DataLoader import CustomMidiDataset

# Get cpu of gpu device for training
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# HYPERPARAMETERS
batch_size = 64
num_epochs = 10000
learning_rate = 1e-3 * 2


# Define Model
class CNNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder using convolutional layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(4, 4)),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.Conv2d(32, 64, kernel_size=(2, 1)),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.Conv2d(64, 256, kernel_size=(2, 1)),
            nn.BatchNorm2d(256),
        )

        # Decoder using transposed convolutional layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=(2, 1)),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.ConvTranspose2d(64, 32, kernel_size=(2, 1)),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.ConvTranspose2d(32, 1, kernel_size=(4, 4)),
            nn.Sigmoid()
        )

    def bottleneck(self, x):
        """
        Create the latent representation
        :param x: The input tensor
        :return: The latent representation
        """
        mu = x.clone()
        logvar = x.clone()
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Create latent representation by reparametrising the input
        :param mu: The mean
        :param logvar:
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        ## encode ##
        x = self.encoder(x)

        # Bottleneck in centre of network
        z, mu, logvar = self.bottleneck(x)
        x = z

        ## decode ##
        x = self.decoder(x)

        return x


def train_loop(train_loader, val_loader, model):
    size = len(train_loader.dataset)
    optimizer = optim.Adam(model.parameters(), learning_rate)

    # Run on val set every n epochs
    every_n_run_val = 10
    # Define Loss Function (choice of 4 different types)
    #loss_fn = nn.CrossEntropyLoss()
    #loss_fn = nn.L1Loss()
    loss_fn = nn.MSELoss()
    #loss_fn = nn.KLDivLoss()

    for epoch in range(num_epochs):
        # Switch to training
        model.train()
        for batch, data in enumerate(train_loader):
            optimizer.zero_grad()
            X = data['input']
            y = data['label']
            num_rows = X.shape[1]
            num_columns = X.shape[2]

            # Add extra dimension for picky CNN and reshape to (BATCH_SIZE, NUM_CHANNELS, WIDTH, HEIGHT)
            X = torch.unsqueeze(X, 0).reshape(batch_size, 1, num_rows, num_columns)
            y = torch.unsqueeze(y, 0).reshape(batch_size, 1, num_rows, num_columns)
            # Compute prediction error and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()

        # Print at end of epoch
        loss, current = loss.item(), batch * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        # Every n epochs or iterations or whatever
        if epoch % every_n_run_val == 0:
            # Do test loop like
            model.eval()
            losses = []
            for batch, data in enumerate(val_loader):
                X = data['input']
                y = data['label']
                num_rows = X.shape[1]
                num_columns = X.shape[2]

                # Add extra dimension for picky CNN and reshape to (BATCH_SIZE, NUM_CHANNELS, WIDTH, HEIGHT)
                X = torch.unsqueeze(X, 0).reshape(batch_size, 1, num_rows, num_columns)
                y = torch.unsqueeze(y, 0).reshape(batch_size, 1, num_rows, num_columns)
                # Compute prediction error and loss
                pred = model(X)
                loss = loss_fn(pred, y)
                if device == 'cuda':
                    losses.append(loss.cpu().detach().numpy())
                else:
                    losses.append(loss.detach().numpy())

            avg_eval_loss = np.mean(losses)
            print("Avg. validation losses: ", avg_eval_loss)

            model.train()


if __name__ == '__main__':
    model = CNNet()
    # Create an instance of the NeuralNetwork and move it to device
    model = model.to(device)
    dataset = CustomMidiDataset()

    # Train-validation split (0.1 means 90% training, 10% validation data)
    validation_split = 0.2
    # Whether to shuffle the data before training
    shuffle_dataset = True
    # Seed for randomly shuffling dataset
    random_seed = 42

    # Create data indices for training and validation splits
    dataset_size = len(dataset)
    print("Dataset length: ", dataset_size, " segments.")
    indices = list(range(dataset_size))
    # Get split index
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating data samplers and loaders
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               sampler=train_sampler,
                                               drop_last=True)
    validation_loader = torch.utils.data.DataLoader(dataset,
                                                    batch_size=batch_size,
                                                    sampler=valid_sampler,
                                                    drop_last=True)

    train_loop(train_loader, validation_loader, model)

    # Save model weights when trained
    torch.save(model.state_dict(), "ANALYSIS/TRAINING/2ND TRAIN/my_trained_model.pt")
