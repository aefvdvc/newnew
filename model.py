import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class GMM(nn.Module):
    def __init__(self, n_components, latent_dim):
        super(GMM, self).__init__()
        self.n_components = n_components
        self.latent_dim = latent_dim
        # 将混合权重 pi 转换为可训练参数
        self.pi = nn.Parameter(torch.ones(n_components) / n_components)
        # 初始化均值和对数方差为可训练的参数
        self.mu = nn.Parameter(torch.randn(n_components, latent_dim))
        self.log_var = nn.Parameter(torch.zeros(n_components, latent_dim))

    def sample(self, batch_size):
        cat = torch.distributions.Categorical(F.softmax(self.pi, dim=0))  # 使用softmax保证pi是有效的
        comp = cat.sample((batch_size,))
        mu = self.mu[comp]
        std = torch.exp(0.5 * self.log_var[comp])
        eps = torch.randn_like(std)
        return mu + eps * std, comp

    def kl_loss(self, z):
        z = z.unsqueeze(1)
        mu = self.mu.unsqueeze(0)
        log_var = self.log_var.unsqueeze(0)

        # 使用 log_softmax 提升数值稳定性
        log_pi = F.log_softmax(self.pi, dim=0)
        log_prob = -0.5 * (log_var + (z - mu) ** 2 / torch.exp(log_var) + np.log(2 * np.pi))
        log_prob = log_prob.sum(-1)
        log_prob = log_prob + log_pi

        log_sum_exp = torch.logsumexp(log_prob, dim=1, keepdim=False)
        return -log_sum_exp.mean()

    def predict(self, z):
        with torch.no_grad():
            z = z.unsqueeze(1)
            mu = self.mu.unsqueeze(0)
            log_var = self.log_var.unsqueeze(0)

            log_prob = -0.5 * (log_var + (z - mu) ** 2 / torch.exp(log_var) + np.log(2 * np.pi))
            log_prob = log_prob.sum(-1)
            log_prob = log_prob + torch.log(self.pi + 1e-10)
            return torch.argmax(log_prob, dim=1)


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, num_layers=10, hidden_dim=512):
        super(Encoder, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(negative_slope=0.2)]
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU(negative_slope=0.2))
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, num_layers=10, hidden_dim=256):
        super(Decoder, self).__init__()
        layers = [nn.Linear(latent_dim, hidden_dim), nn.LeakyReLU(negative_slope=0.2)]
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU(negative_slope=0.2))
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, z):
        return self.model(z)


class RoundtripModel(nn.Module):
    def __init__(self, input_dim, latent_dim, n_components):
        super(RoundtripModel, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        self.gmm = GMM(n_components, latent_dim)

    def forward(self, x):
        z_enc = self.encoder(x)
        x_recon = self.decoder(z_enc)
        z_noise, _ = self.gmm.sample(x.size(0))
        z_noise = z_noise.to(x.device)
        x_gen = self.decoder(z_noise)
        z_recon = self.encoder(x_gen)
        return z_enc, x_recon, z_noise, z_recon


def compute_roundtrip_gmm_loss(x, model):
    z_enc, x_recon, z_noise, z_recon = model(x)
    loss_x = F.mse_loss(x_recon, x)  # 重构损失
    loss_kl = model.gmm.kl_loss(z_enc)  # KL散度损失
    loss_rt = F.mse_loss(z_recon, z_noise)  # 往返损失
    return loss_x + loss_kl + loss_rt, loss_x, loss_kl, loss_rt


def visualize_latent_space(model, data_loader, device):
    model.eval()
    zs = []
    labels = []
    with torch.no_grad():
        for batch, (x, y) in enumerate(data_loader):
            x = x.to(device)
            z = model.encoder(x)
            zs.append(z.cpu())
            labels.append(y)
            if batch > 20:
                break
    zs = torch.cat(zs).numpy()
    labels = torch.cat(labels).numpy()
    tsne = TSNE(n_components=2, random_state=42)
    z_tsne = tsne.fit_transform(zs)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.legend(*scatter.legend_elements(), title="Digits")
    plt.title("Latent Space Visualization via t-SNE")
    plt.savefig("latent_space_tsne.png")
    plt.show()
