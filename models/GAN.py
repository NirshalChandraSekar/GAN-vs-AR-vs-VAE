import torch
import torch.nn as nn
from torch.autograd import Variable

class Discriminator(nn.Module):
    def __init__(self, channels = [256, 128, 64]):
        super().__init__()
        
        ##################
        # TODO: Implement the Discriminator architecture
        # Hint: use three fully connected layers with LeakyReLU activations
        #       and dropout as a form of regularization. The final output 
        #       should be a single logit (no Sigmoid).
        ##################
        self.model = nn.Sequential(
            # TODO: fill in layers here
            nn.Linear(784, channels[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),

            nn.Linear(channels[0], channels[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),

            nn.Linear(channels[1], channels[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),

            nn.Linear(channels[2], 1)
        )

    def forward(self, x):
        x = x.view(x.size(0), 784)
        out = self.model(x)
        out = out.view(out.size(0), -1)
        return out

class Generator(nn.Module):
    def __init__(self, dim_z=100, 
                #  channels = [64, 128, 256]
                # channels = [128, 256, 512]
                channels = [64, 126, 256, 512]
                 ):
        super().__init__()
        self.dim_z = dim_z

        ##################
        # TODO: Implement the Generator architecture
        # Hint: use three fully connected layers with LeakyReLU activations.
        #       The final output layer should map to 784 and use Tanh.
        ##################
        self.model = nn.Sequential(
            # TODO: fill in layers here
            nn.Linear(self.dim_z, channels[0]),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(channels[0], channels[1]),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(channels[1], channels[2]),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(channels[2], channels[3]),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(channels[3], 784),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(x.size(0), self.dim_z)
        out = self.model(x)
        return out


def train_discriminator(discriminator, d_optimizer, images, fake_images, criterion, device):
    
    discriminator.zero_grad()
    outputs_real = discriminator(images)

    ############################
    # TODO: Create labels
    # Hint: real_labels should be all 1s, fake_labels all 0s
    ############################
    real_labels = torch.ones(outputs_real.size(0), 1).to(device)

    ############################
    # TODO: Compute loss on real images
    ############################
    real_loss = criterion(outputs_real, real_labels)

    outputs_fake = discriminator(fake_images.detach())
    fake_labels = torch.zeros(outputs_fake.size(0), 1).to(device)


    ############################
    # TODO: Compute loss on fake images
    ############################
    fake_loss = criterion(outputs_fake, fake_labels)

    ############################
    # TODO: Combine real and fake loss
    ############################
    d_loss = real_loss + fake_loss
    
    d_loss.backward()
    d_optimizer.step()

    return d_loss, outputs_real, outputs_fake, fake_loss

def train_generator(generator, g_optimizer, discriminator_outputs, criterion, device):
    
    generator.zero_grad()

    ############################
    # TODO: Generator wants to fool the discriminator
    # Hint: use real_labels = all 1s (pretend fake images are real)
    ############################
    real_labels = torch.ones(discriminator_outputs.size(0), 1).to(device)

    ############################
    # TODO: Compute generator loss
    # Hint: criterion(discriminator_outputs, real_labels)
    ############################
    g_loss = criterion(discriminator_outputs, real_labels)

    g_loss_fake = criterion(discriminator_outputs, 1 - real_labels.view(-1, 1))
    
    g_loss.backward()
    g_optimizer.step()

    return g_loss, g_loss_fake