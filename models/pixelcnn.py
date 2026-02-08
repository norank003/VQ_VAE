
from MaskedCNN import MaskedCNN
import torch.nn as nn

class PixelCNN(nn.Module):    # Codebook from encoder as grid 7X7 to emmbeding ?? 
  def __init__(self,img_indcies:int ,num_embeddings:int, embedding_dim:int , hidden_dim:int) :
     super().__init__()
     self.img_indcies=img_indcies

     self.embedding = nn.Embedding(num_embeddings, embedding_dim)
     self.conv_in = MaskedCNN('A', embedding_dim, hidden_dim, kernel_size=7, padding=3)
     self.blocks = nn.Sequential
        

     self.conv_out = nn.Conv2d(hidden_dim, num_embeddings, kernel_size=1)

  def forward(self, x):
        # x shape: (B, 7, 7) - The grid of indices
        
        # Step 1: Embed (B, 7, 7) -> (B, 7, 7, embedding_dim)
        x = self.embedding(x)
        
        # Step 2: Permute for Conv2d (B, C, H, W) -> (B, embedding_dim, 7, 7)
        x = x.permute(0, 3, 1, 2)
        
        # Step 3: Pass through Masked Convolutions
        x = F.relu(self.conv_in(x))
        x = self.blocks(x)
        
        # Step 4: Final Logits (B, 512, 7, 7)
        return self.conv_out(x)   

