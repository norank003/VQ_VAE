import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1),
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder_Block(nn.Module):
    def __init__(self, input_channels: int, hidden_dims=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dims // 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dims // 2, hidden_dims, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dims, hidden_dims, 3, padding=1),
        )

    def forward(self, x):
        return self.encoder(x)


class Vector_Quantizer_Block(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=256, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(
            -1.0 / num_embeddings, 1.0 / num_embeddings
        )

    @property
    def e_i_ts(self):
        """
        Codebook tensor expected by PixelCNN
        Shape: (embedding_dim, num_embeddings)
        """
        return self.embeddings.weight.t()

    def forward(self, x):
        # BCHW -> BHWC
        x_perm = x.permute(0, 2, 3, 1).contiguous()
        flat_x = x_perm.view(-1, self.embedding_dim)

        # Compute distances
        distances = (
            torch.sum(flat_x ** 2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_x, self.embeddings.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embeddings(encoding_indices).view(x_perm.shape)

        # Losses
        dictionary_loss = F.mse_loss(quantized.detach(), x_perm)
        commitment_loss = F.mse_loss(quantized, x_perm.detach())

        # Back to BCHW
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        # Straight-through estimator
        quantized = x + (quantized - x).detach()

        return quantized, dictionary_loss, commitment_loss, encoding_indices.view(
            x.shape[0], x.shape[2], x.shape[3]
        )


class Decoder_Block(nn.Module):
    def __init__(self, hidden_dims, out_channels):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dims, hidden_dims, 3, padding=1),
            Residual_Block(hidden_dims),
            Residual_Block(hidden_dims),
            nn.ConvTranspose2d(hidden_dims, hidden_dims, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dims, out_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(x)


class VQ_VAE_Model(nn.Module):
    def __init__(self, input_channels=1, num_embeddings=512, embedding_dim=256):
        super().__init__()
        self.encoder = Encoder_Block(input_channels, embedding_dim)
        self.vq_layer = Vector_Quantizer_Block(num_embeddings, embedding_dim)
        self.decoder = Decoder_Block(embedding_dim, input_channels)

    def encode(self, x):
        """
        Used during PixelCNN training.
        Returns quantized output + indices.
        """
        z = self.encoder(x)
        return self.vq_layer(z)

    def decode_indices(self, indices):
        """
        (B, 7, 7) -> (B, 1, 28, 28)
        """
        # (B, 7, 7) -> (B, 7, 7, 256)
        z_q = F.embedding(indices, self.vq_layer.e_i_ts.t())

        # -> (B, 256, 7, 7)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return self.decoder(z_q)

    def forward(self, x):
        z = self.encoder(x)
        z_q, d_loss, c_loss, indices = self.vq_layer(z)
        x_hat = self.decoder(z_q)
        return x_hat, d_loss, c_loss, indices
