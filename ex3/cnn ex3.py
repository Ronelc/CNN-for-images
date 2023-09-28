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
    return train_dataset, test_dataset, train_loader, test_loader


def plot_loss(g_loss, d_loss, title):
    # Plot the losses
    steps = np.arange(len(g_loss))
    plt.plot(steps, g_loss, label='Generator Loss')
    plt.plot(steps, d_loss, label='Discriminator Loss')

    # Set plot labels and title
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title(f'Generator and Discriminator Losses {title} Loss')

    # Set legend
    plt.legend()

    # Display the plot
    plt.show()


##############################################################################
# Generator and discriminator.
##############################################################################


# Define the Generator network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.convTranspose1 = nn.ConvTranspose2d(100, 64 * 4, 3, 2, 0,
                                                 bias=False).to(device)
        self.batchNorm1 = nn.BatchNorm2d(64 * 4).to(device)
        self.convTranspose2 = nn.ConvTranspose2d(64 * 4, 64 * 2, 3, 2, 0,
                                                 bias=False).to(device)
        self.batchNorm2 = nn.BatchNorm2d(64 * 2).to(device)
        self.convTranspose3 = nn.ConvTranspose2d(64 * 2, 64, 3, 2, 0,
                                                 bias=False).to(device)
        self.batchNorm3 = nn.BatchNorm2d(64).to(device)
        self.convTranspose4 = nn.ConvTranspose2d(64, 1, 3, 2, 2, 1,
                                                 bias=False).to(device)
        self.relu = nn.ReLU().to(device)
        self.sigmoid = nn.Sigmoid().to(device)

    def forward(self, x):
        x = self.relu(self.batchNorm1(self.convTranspose1(x)))
        x = self.relu(self.batchNorm2(self.convTranspose2(x)))
        x = self.relu(self.batchNorm3(self.convTranspose3(x)))
        x = self.sigmoid(self.convTranspose4(x))
        return x


# Define the Discriminator network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(1, 28, 4, 2, 1, bias=False).to(device)
        self.conv2 = nn.Conv2d(28, 28 * 2, 4, 2, 1, bias=False).to(device)
        self.batchNorm2 = nn.BatchNorm2d(28 * 2).to(device)
        self.conv3 = nn.Conv2d(28 * 2, 28 * 4, 4, 2, 1, bias=False).to(device)
        self.batchNorm3 = nn.BatchNorm2d(28 * 4).to(device)
        self.conv4 = nn.Conv2d(28 * 4, 1, 4, 2, 1, bias=False).to(device)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True).to(device)
        self.sigmoid = nn.Sigmoid().to(device)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.batchNorm2(self.conv2(x)))
        x = self.leaky_relu(self.batchNorm3(self.conv3(x)))
        x = self.sigmoid(self.conv4(x))
        return x


##############################################################################
# Encoder
##############################################################################

class ENCODER(nn.Module):

    def __init__(self, d=100):
        super(ENCODER, self).__init__()
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


##############################################################################
# Train
##############################################################################

def train_GAN_model(generator, discriminator, dataloader, num_epochs,
                    adversarial_loss, learning_rate, latent_dim, saturating):
    # Optimizers
    generator_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
    discriminator_optimizer = optim.Adam(discriminator.parameters(),
                                         lr=learning_rate)

    loss_dict = {'g': [], 'd': []}
    arr = []
    for epoch in range(num_epochs):
        for i, (real_images, labels) in enumerate(dataloader, 0):
            real_images = real_images.to(device)
            # print(real_images.shape)
            batch_size = real_images.size(0)

            # Create labels for real and fake images
            real_labels = torch.ones(batch_size, 1, 1, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1, 1, 1).to(device)

            # Train the discriminator
            discriminator_optimizer.zero_grad()

            # Real images
            real_output = discriminator(real_images)
            real_loss = adversarial_loss(real_output, real_labels)

            # Fake images
            latent_vector = torch.randn((batch_size, latent_dim, 1, 1)).to(
                device)
            generated_images = generator(latent_vector)
            fake_output = discriminator(generated_images.detach())
            fake_loss = adversarial_loss(fake_output, fake_labels)

            # Discriminator loss and backward propagation
            discriminator_loss = real_loss + fake_loss
            discriminator_loss.backward()
            discriminator_optimizer.step()

            # Train the generator
            generator_optimizer.zero_grad()

            # Generate fake images and get discriminator output
            generated_images = generator(latent_vector)
            fake_output = discriminator(generated_images)
            if i % (len(dataloader)) == 0:
                arr.append(
                    generated_images[0].cpu().detach().numpy().reshape(28, 28))

            # Generator loss and backward propagation
            coef = 1 if saturating else -1
            labels = real_labels if saturating else fake_labels
            generator_loss = coef * adversarial_loss(fake_output, labels)
            generator_loss.backward()
            generator_optimizer.step()

            loss_dict['d'].append(discriminator_loss.item())
            loss_dict['g'].append(generator_loss.item())

        # Print training progress
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] | Generator Loss: {generator_loss.item():.4f} | Discriminator Loss: {discriminator_loss.item():.4f}")
    return loss_dict, arr


def train_GAN_Inversion(encoder, generator, num_epochs, train_loader,
                        criterion=nn.MSELoss()):
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
    loss_arr = []
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            latent = encoder(inputs)
            latent = latent.unsqueeze(2).unsqueeze(2)
            g_outputs = generator(latent)
            loss = criterion(inputs, g_outputs)
            loss.backward()
            optimizer.step()

            loss_arr.append(loss.item())
            if i % len(train_loader) == 0:
                # plot input image and generator output
                fig, axes = plt.subplots(1, 2, figsize=(6, 3))
                axes[0].imshow(inputs[0].cpu().reshape(28, 28), cmap='gray')
                axes[0].set_title('Input image')
                axes[1].imshow(
                    g_outputs[0].cpu().detach().numpy().reshape(28, 28),
                    cmap='gray')
                axes[1].set_title('Generator output')
                for ax in axes:
                    ax.axis('off')
                plt.tight_layout()
                plt.show()

    print('Finished Training')
    return loss_arr


def train_noisy_model(encoder, generator, num_epochs, train_loader, criterion):
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)
    loss_arr = []
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        for i, input_im in enumerate(train_loader, 0):
            input_im = input_im.to(device)
            img_to_show = input_im
            input_im = input_im.unsqueeze(1)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            latent = encoder(input_im)
            latent = latent.unsqueeze(2).unsqueeze(2)
            g_outputs = generator(latent)
            loss = criterion(input_im, g_outputs)
            loss.backward()
            optimizer.step()

            loss_arr.append(loss.item())
            if i % len(train_loader) / 2 == 0:
                # plot input image and generator output
                fig, axes = plt.subplots(1, 2, figsize=(6, 3))
                axes[0].imshow(img_to_show[0].cpu().reshape(28, 28),
                               cmap='gray')
                axes[0].set_title('Input image')
                axes[1].imshow(
                    g_outputs[0].cpu().detach().numpy().reshape(28, 28),
                    cmap='gray')
                axes[1].set_title('Generator output')
                for ax in axes:
                    ax.axis('off')
                plt.tight_layout()
                plt.show()
    print('Finished Training')
    return loss_arr


##############################################################################
# questions
##############################################################################

latent_dim_ = 100
batch_size_ = 128
num_epochs_ = 20
lr_ = 0.0002
train_dataset_, test_dataset_, train_loader_, test_loader_ = create_data_set(
    batch_size_)

##############################################################################
# question 1 - Loss Saturation.
##############################################################################

for loss_, text in [(nn.BCELoss(), 'BCE'),
                    (nn.BCELoss(), 'BCE non saturating'),
                    (nn.MSELoss(), 'MSE')]:
    saturating_ = text != 'BCE non saturating'
    print(f'Start train with {text} Loss')
    G, D = Generator(), Discriminator()
    loss_dict_, arr_ = train_GAN_model(G, D, train_loader_, num_epochs_, loss_,
                                       lr_, latent_dim_, saturating_)
    plot_loss(loss_dict_['g'], loss_dict_['d'], text)

    # plot G images
    concatenated_image = np.concatenate(arr_, axis=1)
    fig_, ax_ = plt.subplots(figsize=(20, 20))
    ax_.imshow(concatenated_image, cmap='gray')
    ax_.axis('off')
    plt.show()

##############################################################################
# question 2 - Model Inversion.
##############################################################################

G, D, encoder_ = Generator(), Discriminator(), ENCODER()
print('Start generator train')
train_GAN_model(G, D, train_loader_, num_epochs_, nn.BCELoss(), lr_,
                latent_dim_, False)
print('Finish to train generator')
loss_arr_ = train_GAN_Inversion(encoder_, G, num_epochs_, train_loader_)
plt.plot(loss_arr_)
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.show()

##############################################################################
# question 3 - Image restoration.
##############################################################################

# 3.1 - restoration of noisy images:
G, D, encoder_ = Generator(), Discriminator(), ENCODER()
print('Start generator train')
train_GAN_model(G, D, train_loader_, num_epochs_, nn.BCELoss(), lr_,
                latent_dim_, False)
print('Finish generator train')


# Add i.i.d. Normal noise to the MNIST dataset
train_dataset_ = train_dataset_.data.float() / 255.0
std_dev = 0.1
noise = torch.randn_like(train_dataset_) * std_dev
noisy_images = torch.clamp(train_dataset_ + noise, min=0, max=1)

noisy_images_loader = DataLoader(dataset=noisy_images, batch_size=batch_size_,
                              shuffle=True, num_workers=2)

loss_arr_ = train_noisy_model(encoder_, G, num_epochs_, noisy_images_loader, nn.MSELoss())
plt.plot(loss_arr_)
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.show()


# 3.2 - inpainting.
import random

G, D, encoder_ = Generator(), Discriminator(), ENCODER()
print('Start generator train')
train_GAN_model(G, D, train_loader_, num_epochs_, nn.BCELoss(), lr_,
                latent_dim_, False)
print('Finish generator train')

train_dataset_ = test_dataset_.data.float() / 255.0

window_size = 8

# Delete random windows from the MNIST dataset
deleted_mnist_data = train_dataset_.clone()

for i in range(len(deleted_mnist_data)):
    # Generate random coordinates for the window
    x = random.randint(0, train_dataset_.size(2) - window_size)
    y = random.randint(0, train_dataset_.size(1) - window_size)

    # Set the window region to zeros
    deleted_mnist_data[i, y:y + window_size, x:x + window_size] = 0.0

noisy_images_loader = DataLoader(dataset=deleted_mnist_data,
                                 batch_size=batch_size_,
                                 shuffle=True, num_workers=2)

loss_arr_ = train_noisy_model(encoder_, G, num_epochs_, noisy_images_loader,
                              nn.L1Loss())
plt.plot(loss_arr_)
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.show()
