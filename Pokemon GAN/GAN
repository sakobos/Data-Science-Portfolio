import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, ToPILImage
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

# Image transformations to create tensors w/ size (256,256) and normalize image pixel values to [-1,1]
transform = Compose([ToTensor(), Resize((64, 64), antialias=True), Normalize(mean=(.5, .5, .5), std=(.5, .5, .5))])

# Path for Gen 1 Dataset
pokemon_path = '/Users/skobos/Documents/Data Science Portfolio/Pokemon GAN/PokemonData/Charizard'

# Create an ImageFolder dataset
dataset = ImageFolder(root=pokemon_path, transform=transform)

# Choose a random index to show a random normalized vs denormalized image
random_index = np.random.randint(0, len(dataset))
# Get the normalized and denormalized images
random_img_normalized, random_label = dataset[random_index]
# Denormalizing the image
random_img_denormalized = random_img_normalized * 0.5 + 0.5
# Clip the denormalized image values to the valid range [0, 1]
random_img_denormalized = torch.clamp(random_img_denormalized, 0, 1)
# Converting the images to PIL to display
random_img_normalized_pil = ToPILImage()(random_img_normalized)
random_img_denormalized_pil = ToPILImage()(random_img_denormalized)
# Showing images side-by-side
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# Getting normalized image
axes[0].imshow(random_img_normalized_pil)
axes[0].set_title(f'Normalized\nClass: {random_label}')
# Getting denormalized image
axes[1].imshow(random_img_denormalized_pil)
axes[1].set_title(f'Denormalized\nClass: {random_label}')
plt.show()

# Create a DataLoader with the transformed dataset
# Batch size of 110, so we have no "remainder" batch that isn't full
batch_size = 7
# 5 batches
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Designing the GAN: Going for DCGAN Structure
image_channels = 3
image_size = 12288  # 64x64
hidden_size = 128
latent_size = 128
LR = .0001
LRLSlope = .1
kernel = (3, 3)
stride = 2
padding = 1
input_size = 64

# Define the Discriminator
Discriminator = nn.Sequential(
    nn.Conv2d(3, hidden_size, kernel, stride, padding),
    nn.BatchNorm2d(hidden_size),
    nn.LeakyReLU(LRLSlope),
    nn.Conv2d(hidden_size, hidden_size, kernel, stride, padding),
    nn.BatchNorm2d(hidden_size),
    nn.LeakyReLU(LRLSlope),
    nn.Conv2d(hidden_size, 1, kernel, stride, padding),
    nn.AdaptiveAvgPool2d(1),  # Global average pooling
    nn.Flatten(),  # flatten output before final fully connected layer
    nn.Sigmoid()
)


# Define the Generator
Generator = nn.Sequential(
    nn.ConvTranspose2d(latent_size, hidden_size * 8, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(hidden_size * 8),
    nn.ReLU(True),
    nn.ConvTranspose2d(hidden_size * 8, hidden_size * 4, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(hidden_size * 4),
    nn.ReLU(True),
    nn.ConvTranspose2d(hidden_size * 4, hidden_size * 2, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(hidden_size * 2),
    nn.ReLU(True),
    nn.ConvTranspose2d(hidden_size * 2, hidden_size, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(hidden_size),
    nn.ReLU(True),
    nn.ConvTranspose2d(hidden_size, 3, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
)

loss_fn = nn.BCELoss()
g_optimizer = torch.optim.Adam(Generator.parameters(), LR)
d_optimizer = torch.optim.Adam(Discriminator.parameters(), LR)


def resetting_gradient():
    g_optimizer.zero_grad()
    d_optimizer.zero_grad()


def discriminator_training(images):
    batch_size = images.size(0)
    real_labels = torch.ones(batch_size, 1)
    fake_labels = torch.zeros(batch_size, 1)

    # Real Image Loss
    d_out_real = Discriminator(images)
    # print(f"Real Images - d_out_real size: {d_out_real.size()}")
    d_out_real = d_out_real.view(-1, 1)  # Flatten the output tensor
    d_real_loss = loss_fn(d_out_real, real_labels)

    # Fake Image Loss
    rand_tensor = torch.randn(batch_size, latent_size, 1, 1, device=images.device)
    fake_images = Generator(rand_tensor)
    d_out_fake = Discriminator(fake_images.detach())
    # print(f"Fake Images - d_out_fake size: {d_out_fake.size()}")
    d_out_fake = d_out_fake.view(-1, 1)  # Flatten the output tensor
    d_fake_loss = loss_fn(d_out_fake, fake_labels)

    # Combine Real & Fake Loss
    d_total_loss = d_real_loss + d_fake_loss

    # Reset Gradients
    resetting_gradient()
    # Compute Gradients
    d_total_loss.backward()
    # Backpropagation
    d_optimizer.step()

    return d_total_loss


def generator_training():
    rand_tensor = torch.randn(batch_size, latent_size, 1, 1)
    fake_images = Generator(rand_tensor)
    labels = torch.ones(batch_size, 1)
    g_loss = loss_fn(Discriminator(fake_images), labels)

    # Reset Gradients
    resetting_gradient()
    # Compute Gradients
    g_loss.backward()
    # Backpropagation

    return g_loss, fake_images


# Training the GAN
epochs = 100
steps = len(dataloader)
d_total_loss, g_loss = [], []

for epoch in range(epochs):
    for i, (images, _) in enumerate(dataloader):
        # Load a batch & transform to vectors
        # images = images.reshape(batch_size, -1)

        # Train the discriminator and generator
        d_total_loss = discriminator_training(images)
        g_loss, fake_images = generator_training()
# Inspect the losses
        if (i + 1) % 2 == 0:
            # d_total_loss.append(d_total_loss.item())
            # g_loss.append(g_loss.item())
            # real_scores.append(real_score.mean().item())
            # fake_scores.append(fake_score.mean().item())
            print('Epoch [{}/{}], Step [{}/{}]' # , d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                  .format(epoch, epochs, i + 1, steps, d_total_loss.item(), g_loss.item()))
                         # real_score.mean().item(), fake_score.mean().item()))


def denorm(x):
    # Inverse of the Normalize transform
    inv_normalize = Normalize(mean=(-0.5, -0.5, -0.5), std=(2, 2, 2))
    # Move to CPU and detach from the computation graph
    x_cpu = x.detach().cpu()
    # Denormalize each image in the batch
    denormalized = torch.stack([inv_normalize(img.reshape(3, 64, 64)) for img in x_cpu])
    # Convert to NumPy array
    denormalized = denormalized.numpy()
    return denormalized


# Display the final generated images
sample_vectors = torch.randn(batch_size, latent_size, 1, 1)
final_fake_images = Generator(sample_vectors)
final_fake_images = final_fake_images.reshape(final_fake_images.size(0), 3, 64, 64)
final_fake_images = denorm(final_fake_images.detach())

# Plot the final generated images
fig, axes = plt.subplots(1, min(10, len(final_fake_images)), figsize=(15, 5))
for i in range(min(10, len(final_fake_images))):
    # Transpose the dimensions and normalize the image data
    img = np.transpose(final_fake_images[i], (1, 2, 0)) / 255.0
    axes[i].imshow(img)
    axes[i].axis('off')
plt.show()


