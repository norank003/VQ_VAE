
from models.maskedcnn import MaskedCNN
import torch.nn.functional as F
import torch.nn as nn

class PixelCNNResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 1),
            nn.ReLU(True),
            MaskedCNN('B', dim // 2, dim // 2, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(dim // 2, dim, 1)
        )

    def forward(self, x):
        return x + self.block(x)

class PixelCNN(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        # Initial Mask Type A
        self.conv_in = MaskedCNN('A', embedding_dim, hidden_dim, kernel_size=7, padding=3)

        # Stack 8-12 Residual Blocks
        self.layers = nn.ModuleList([PixelCNNResBlock(hidden_dim) for _ in range(10)])

        self.conv_out = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, 512, 1), # First 1x1 to refine
            nn.ReLU(True),
            nn.Conv2d(512, num_embeddings, 1) # Final prediction
        )

    def forward(self, x):
        x = self.embedding(x).permute(0, 3, 1, 2)
        x = self.conv_in(x)
        for layer in self.layers:
            x = layer(x)
        return self.conv_out(x)


