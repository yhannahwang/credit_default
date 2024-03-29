import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim

data_path = os.getcwd() + '/data/credit_card_default.csv'
df_credit = pd.read_csv(data_path,index_col='ID')

X_train = df_credit.values

# Define the indices of columns where you want to apply Winsorization
columns_to_winsorize = [i for i in range(11,17)]  # Adjust according to your specific columns

# Calculate the percentiles for the columns to be Winsorized
p99 = np.percentile(X_train[:, columns_to_winsorize], 99, axis=0)
p1 = np.percentile(X_train[:, columns_to_winsorize], 1, axis=0)

# Apply Winsorization to the selected columns
X_train[:, columns_to_winsorize] = np.clip(X_train[:, columns_to_winsorize], p1, p99)

# Initialize MinMaxScaler
scaler = MinMaxScaler()
# Fit the scaler
scaler.fit(X_train)

# Transform your data
X_train_normalized = scaler.transform(X_train)
X_train_tensor = torch.tensor(X_train_normalized, dtype=torch.float32)

# Define the Generator network
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        # Define generator architecture
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.ReLU(),
            nn.Linear(12, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.fc(x)


# Define the Discriminator network
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        # Define discriminator architecture
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.ReLU(),
            nn.Linear(12, 1),
            nn.ReLU()  # Output a probability between 0 and 1
        )

    def forward(self, x):
        return self.fc(x)


# Define the GAN
class GAN:
    def __init__(self, generator, discriminator, data_dim):
        self.generator = generator
        self.discriminator = discriminator
        self.data_dim = data_dim
        self.loss_function = nn.BCELoss()  # Binary cross-entropy loss
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=0.0001)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0001)

    def train(self, real_data, num_epochs):
        for epoch in range(num_epochs):
            # Train discriminator
            self.discriminator_optimizer.zero_grad()
            real_output = self.discriminator(real_data)
            real_loss = self.loss_function(real_output, torch.ones(real_data.size(0), 1))

            noise = torch.randn(real_data.size(0), self.data_dim)
            fake_data = self.generator(noise)
            fake_output = self.discriminator(fake_data.detach())
            fake_loss = self.loss_function(fake_output, torch.zeros(real_data.size(0), 1))

            discriminator_loss = real_loss + fake_loss
            discriminator_loss.backward()
            ## clip
            disc_params = [param for param in discriminator.parameters() if param.grad is not None]
            disc_grad_norm = torch.nn.utils.clip_grad_norm_(disc_params, max_norm=0.2)
            ##
            self.discriminator_optimizer.step()

            # Train generator
            self.generator_optimizer.zero_grad()
            noise = torch.randn(real_data.size(0), self.data_dim)
            fake_data = self.generator(noise)
            fake_output = self.discriminator(fake_data)
            generator_loss = self.loss_function(fake_output, torch.ones(real_data.size(0), 1))
            generator_loss.backward()
            ##
            gen_params = [param for param in generator.parameters() if param.grad is not None]
            gen_grad_norm = torch.nn.utils.clip_grad_norm_(gen_params, max_norm=0.2)
            ##
            self.generator_optimizer.step()

    def generate_samples(self, num_samples):
        noise = torch.randn(num_samples, self.data_dim)
        fake_data = self.generator(noise)
        return fake_data.detach().numpy()



# Assuming 'X_train_tensor' is a PyTorch tensor containing preprocessed data
input_dim = X_train_tensor.shape[1]  # Dimensionality of input noise vector
output_dim = X_train_tensor.shape[1]  # Dimensionality of output data

# Create generator and discriminator instances
generator = Generator(input_dim, output_dim)
discriminator = Discriminator(output_dim)

# Define GAN instance
gan = GAN(generator, discriminator, input_dim)

# Train GAN
num_epochs = 100
gan.train(X_train_tensor, num_epochs)

# Generate synthetic data
num_samples = 5000  # Adjust as needed
synthetic_data = gan.generate_samples(num_samples)
synthetic_data_scale = scaler.inverse_transform(synthetic_data)
# Convert synthetic data to DataFrame if needed
synthetic_df = pd.DataFrame(synthetic_data_scale, columns=df_credit.columns)
rounded_df = synthetic_df.round(decimals=0)
rounded_df.to_csv(os.getcwd() +'/gan.csv', index=False)

