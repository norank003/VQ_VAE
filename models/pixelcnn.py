
from models.maskedcnn import MaskedCNN
import torch.nn.functional as F
import torch.nn as nn

class PixelCNN(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        
        # Initial Masked A layer
        self.conv_in = MaskedCNN('A', embedding_dim, hidden_dim, kernel_size=7, padding=3)
        
        # Increase to 6 Residual Blocks for better receptive field
        self.blocks = nn.ModuleList([
            nn.Sequential(
                MaskedCNN('B', hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, hidden_dim, 1) # Optional 1x1 conv to refine features
            ) for _ in range(6)
        ])
        
        self.conv_out = nn.Conv2d(hidden_dim, num_embeddings, kernel_size=1)

    def forward(self, x):
        x = self.embedding(x).permute(0, 3, 1, 2)
        x = F.relu(self.conv_in(x))
        
        # Applying Residual Connections
        for block in self.blocks:
            x = x + block(x) # Residual: current + transformation
            x = F.relu(x)
            
        return self.conv_out(x)

