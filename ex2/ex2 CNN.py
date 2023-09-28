import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

# Set device to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


##############################################################################
# encoders and decoders
##############################################################################


class ENCODER_1(nn.Module):

    def __init__(self, d):
        super(ENCODER_1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5,
                               stride=2, padding=1).to(device)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=d // 2,
                               kernel_size=5, stride=2, padding=1).to(device)
        self.conv3 = nn.Conv2d(in_channels=d // 2, out_channels=d,
                               kernel_size=6).to(device)
        self.flatten = nn.Flatten().to(device)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        return x


class DECODER_1(nn.Module):

    def __init__(self, d):
        super(DECODER_1, self).__init__()
        self.d = d
        self.conv1 = nn.ConvTranspose2d(in_channels=d, out_channels=d // 2,
                                        kernel_size=5, stride=2,
                                        output_padding=1).to(device)
        self.conv2 = nn.ConvTranspose2d(in_channels=d // 2, out_channels=16,
                                        kernel_size=5, stride=2,
                                        output_padding=1).to(device)
        self.conv3 = nn.ConvTranspose2d(in_channels=16, out_channels=1,
                                        kernel_size=13).to(device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), self.d, 2, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.sigmoid(x)
        return x


class ENCODER_2(nn.Module):
    def __init__(self, d):
        super(ENCODER_2, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1).to(
            device)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1).to(
            device)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=7).to(device)
        self.fc = nn.Linear(64, d).to(device)
        self.relu = nn.ReLU().to(device)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class DECODER_2(nn.Module):
    def __init__(self, d):
        super(DECODER_2, self).__init__()
        self.fc = nn.Linear(d, 64).to(device)
        self.conv1 = nn.ConvTranspose2d(64, 32, kernel_size=7).to(device)
        self.conv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2,
                                        padding=1, output_padding=1).to(
            device)
        self.conv3 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2,
                                        padding=1, output_padding=1).to(
            device)
        self.relu = nn.ReLU().to(device)
        self.sigmoid = nn.Sigmoid().to(device)

    def forward(self, x):
        x = self.relu(self.fc(x))
        x = x.view(x.size(0), 64, 1, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.sigmoid(self.conv3(x))
        return x


# Define bigger encoder architecture
class ENCODER_3(nn.Module):

    def __init__(self, d):
        super(ENCODER_3, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5).to(device)
        self.conv2 = nn.Conv2d(6, 10, 5).to(device)
        self.conv3 = nn.Conv2d(10, 16, 5).to(device)
        self.flat = nn.Flatten().to(device)
        self.FC1 = nn.Linear(16 * 16 * 16, 128).to(device)
        self.FC2 = nn.Linear(128, d).to(device)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flat(x)
        x = F.relu(self.FC1(x))
        x = self.FC2(x)
        return x


# Define bigger decoder architecture
class DECODER_3(nn.Module):

    def __init__(self, d):
        super(DECODER_3, self).__init__()
        self.FC1 = nn.Linear(d, 128).to(device)
        self.FC2 = nn.Linear(128, 16 * 16 * 16).to(device)
        self.unflattun = nn.Unflatten(1, (16, 16, 16)).to(device)
        self.conv1 = nn.ConvTranspose2d(16, 10, 5).to(device)
        self.conv2 = nn.ConvTranspose2d(10, 6, 5).to(device)
        self.conv3 = nn.ConvTranspose2d(6, 1, 5).to(device)

    def forward(self, x):
        x = F.relu(self.FC1(x))
        x = F.relu(self.FC2(x))
        x = self.unflattun(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


##############################################################################
# functions
##############################################################################

# Load the MNIST dataset
def create_data_set(batch_size):
    train_dataset = datasets.MNIST(root='data/', train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)
    test_dataset = datasets.MNIST(root='data/', train=False,
                                  transform=transforms.ToTensor(),
                                  download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=2)
    return train_dataset, train_loader, test_loader


def plot_loss(loss_arr, x_axis, title, x_title, d):
    plt.plot(x_axis, loss_arr)
    plt.ylabel("Loss")
    plt.xlabel(x_title)
    plt.title(f"{title}, d: {d}")
    plt.show()


def train_model(num_epoch, d, encoder, decoder, with_pad, train_loader,
                params, is_mse=True, criterion=nn.MSELoss()):
    optimizer = torch.optim.Adam(params, lr=0.001)
    pad = nn.ZeroPad2d(2)

    for epoch in range(num_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, (input, labels) in enumerate(train_loader, 0):
            input, labels = input.to(device), labels.to(device)
            if with_pad:
                input = pad(input)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            latent = encoder(input)
            decoder_outputs = decoder(latent)
            loss = criterion(decoder_outputs,
                             input if is_mse else labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # print every 100 mini-batches
            if i % 100 == 99:
                # show_image(i, d, epoch, running_loss / 100, shape,
                #            decoder_outputs, input)
                running_loss = 0.0
    print(f"end of train, d = {d}")


def show_image(i, d, epoch, running_loss, shape, decoder_outputs, input):
    print('d = ', d)
    print(
        f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss:.3f}')
    print("input:")
    plt.imshow(input[0].cpu().reshape(shape, shape), cmap="gray")
    plt.show()
    print("decoder output:")
    # plt.imshow(
    #     decoder_outputs[0].cpu().detach().numpy().reshape(shape,
    #                                                       shape),
    #     cmap="gray")
    # plt.show()


# test model, and calculate the loss:
def test_model(test_loader, encoder, decoder, with_pad, d, criterion,
               is_mse=True):
    with torch.no_grad():
        test_loss = 0.0
        size = 0
        for i, (input, labels) in enumerate(test_loader, 0):
            input, labels = input.to(device), labels.to(device)
            if with_pad:
                pad = nn.ZeroPad2d(2)
                input = pad(input)
            latent = encoder(input)
            decoder_outputs = decoder(latent)
            test_loss += criterion(decoder_outputs,
                                   input if is_mse else labels).item()
            print("test")
            size += 1
            show_image(i, d=d, epoch=0,
                       running_loss=test_loss / size,
                       shape=32 if with_pad else 28,
                       decoder_outputs=decoder_outputs, input=input)
    return test_loss / len(test_loader)


##############################################################################
# question 1 - find best architecture and best value of d.
##############################################################################
def find_best_arch(num_epoch, train_loader, test_loader):
    d = 15
    loss_array = []
    arch_lst = [(ENCODER_1, DECODER_1, True), (ENCODER_2, DECODER_2, False),
                (ENCODER_3, DECODER_3, False)]
    for enc, dec, pad in arch_lst:
        encoder, decoder = enc(d), dec(d)
        params = [{'params': encoder.parameters()},
                  {'params': decoder.parameters()}]
        train_model(num_epoch, d, encoder, decoder, pad, train_loader, params)
        criterion = nn.MSELoss()
        loss_array.append(
            test_model(test_loader, encoder, decoder, pad, d, criterion))
    plot_loss(loss_array, ['arch1', 'arch2', 'arch_3'],
              'Q1.1 - test loss as a function architecture type',
              "architecture type", d)
    print(loss_array)


def find_best_d(num_epoch, train_loader, test_loader):
    loss_d_arr = []
    for d in range(2, 30, 3):
        encoder, decoder = ENCODER_3(d), DECODER_3(d)
        params = [{'params': encoder.parameters()},
                  {'params': decoder.parameters()}]
        train_model(num_epoch, d, encoder, decoder, False, train_loader,
                    params)
        criterion = nn.MSELoss()
        loss_d_arr.append(
            test_model(test_loader, encoder, decoder, False, d_, criterion))
    plot_loss(loss_d_arr, np.arange(2, 30, 3),
              "Q1.2 - Loss with different values of d", "value of d", "")


##############################################################################
# question 2 - Interpolation.
##############################################################################

def get_two_images(test_loader, pair):
    for i, (images, labels) in enumerate(test_loader, 0):
        img1, img2 = (images[i]).to(device), (images[i + 1]).to(device)
        if (labels[i].item(), labels[i + 1].item()) == pair or (
                labels[i + 1].item(), labels[i].item()) == pair:
            return (
                img1.unsqueeze(0), labels[i], img2.unsqueeze(0), labels[i + 1])


def interpolation(num_epoch, train_loader, test_loader, d):
    encoder, decoder = ENCODER_3(d=d), DECODER_3(d=d)
    params = [{'params': encoder.parameters()},
              {'params': decoder.parameters()}]
    train_model(num_epoch, d, encoder, decoder, False, train_loader, params)
    for pair in [(0, 6), (1, 7), (1, 3)]:
        images_arr = []
        img1, label1, img2, label2 = get_two_images(test_loader, pair)
        print(label1.item(), ' -> ', label2.item())
        for i, alpha in enumerate(np.arange(0, 1, 0.2)):
            img = decoder(
                (encoder(img1) * alpha) + (encoder(img2) * (1 - alpha)))
            images_arr.append(img.cpu().detach().numpy().reshape(28, 28))
        concatenated_image = np.concatenate(images_arr, axis=1)

        # plot the concatenated image
        fig, ax = plt.subplots()
        ax.imshow(concatenated_image, cmap='gray')
        ax.axis('off')
        plt.show()


##############################################################################
# question 3 - Decorrelation
##############################################################################

def decorrelation(d, test_set, train_loader):
    encoder, decoder = ENCODER_3(d=d), DECODER_3(d=d)
    images, labels = next(
        iter(DataLoader(test_set, batch_size=5000, shuffle=True)))
    params = [{'params': encoder.parameters()},
              {'params': decoder.parameters()}]
    train_model(10, d, encoder, decoder, False, train_loader, params)
    latent = encoder(images.to(device))
    correlation = torch.corrcoef(latent.squeeze().T.to(device))
    numerator, denominator = 0, 0
    for i in range(d):
        numerator += torch.sum(torch.abs(correlation[i][i + 1:]))
        denominator += d - i + 1
    return (numerator / denominator).item()


def run_decorrelation(test_set, train_loader):
    correlations = []
    d_arr = np.arange(10, 110, 20)
    for d in d_arr:
        correlations.append(decorrelation(d, test_set, train_loader))
    fig, ax = plt.subplots(figsize=(6, 3))
    plt.plot(d_arr, correlations, marker='o')
    plt.xlabel("value of d")
    plt.ylabel('correlation')
    plt.title('correlation as a function of d')
    for i in range(len(d_arr)):
        ax.text(d_arr[i], correlations[i],
                (d_arr[i], round(correlations[i], 3)), size=12)
    plt.show()


##############################################################################
# question 4 - Transfer Learning.
##############################################################################

def data_for_q4(train_set_size=90):
    batch_size = 10
    train_set = datasets.MNIST(root='./data', train=True,
                               download=True,
                               transform=transforms.ToTensor())
    size = len(train_set) - train_set_size
    q4_train_set = torch.utils.data.random_split(train_set, [train_set_size,
                                                             size])[0]
    train_loader = DataLoader(q4_train_set, batch_size=batch_size,
                              shuffle=True, num_workers=2)
    return train_loader


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.FC1 = nn.Linear(25, 16).to(device)
        self.FC2 = nn.Linear(16, 32).to(device)
        self.FC3 = nn.Linear(32, 10).to(device)

    def forward(self, x):
        x = F.relu(self.FC1(x))
        x = F.relu(self.FC2(x))
        x = self.FC3(x)
        return x


def transfer_learning(d, train_loader, test_loader, parameters, part,
                      pre_trained_encoder, decoder, mlp):
    num_epoch = 10
    q4_train_loader = data_for_q4()

    params = [{'params': pre_trained_encoder.parameters()},
              {'params': decoder.parameters()}]
    train_model(num_epoch, d, pre_trained_encoder, DECODER_3(d), False,
                train_loader, params)

    # train mlp with the pre-trained encoder
    criterion = nn.CrossEntropyLoss()
    train_model(num_epoch, d, pre_trained_encoder, mlp, False,
                q4_train_loader, parameters, is_mse=False,
                criterion=criterion)

    test_loss = test_model(test_loader, pre_trained_encoder, mlp, False, d,
                           criterion, is_mse=False)
    print(f"test Loss of the {part} part: {test_loss}")


##############################################################################
# main
##############################################################################

if __name__ == '__main__':
    batch_size_ = 128
    num_epoch_ = 10

    # create data set
    test_set_, train_loader_, test_loader_ = create_data_set(batch_size_)

    # Q1:

    # Q1.1 - find the best architecture
    find_best_arch(num_epoch_, train_loader_, test_loader_)

    # Q1.2 - test the best architecture with different values of d
    find_best_d(num_epoch_, train_loader_, test_loader_)

    # Q2 - interpolation
    for d_ in [10, 100]:
        print("d: ", d_)
        interpolation(num_epoch_, train_loader_, test_loader_, d_)

    # Q3 - decorrelation
    run_decorrelation(test_set_, train_loader_)

    # Q4 - Transfer Learning.
    d_ = 25

    # Q4.1 - train encoder over all samples for first part
    pre_trained_encoder_, decoder_, mlp_ = ENCODER_3(d_), DECODER_3(d_), MLP()
    mlp_parameters = mlp_.parameters()
    transfer_learning(d_, train_loader_, test_loader_, mlp_parameters, "first",
                      pre_trained_encoder_, decoder_, mlp_)

    # Q4.2 - train encoder over all samples for first part
    pre_trained_encoder_, decoder_, mlp_ = ENCODER_3(d_), DECODER_3(d_), MLP()
    enc_mlp_parameters = [{'params': pre_trained_encoder_.parameters()},
                          {'params': mlp_.parameters()}]
    transfer_learning(d_, train_loader_, test_loader_, enc_mlp_parameters,
                      "second", pre_trained_encoder_, decoder_, mlp_)
