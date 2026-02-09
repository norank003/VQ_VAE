import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from torchvision.utils import save_image, make_grid
from data_setup import DataSetup
from models.vq_vae import VQ_VAE_Model
from models.pixelcnn import PixelCNN

def sample_from_pixelcnn(model, batch_size, device, temperature=1.0):
    """Autoregressively samples a 7x7 grid of indices."""
    model.eval()
    samples = torch.zeros(batch_size, 7, 7).long().to(device)

    with torch.no_grad():
        for i in range(7):
            for j in range(7):
                logits = model(samples) # Output shape: (B, 512, 7, 7)
                # Apply temperature to logits before softmax to control diversity
                probs = F.softmax(logits[:, :, i, j] / temperature, dim=1)
                # Sample from the multinomial distribution
                sampled_indices = torch.multinomial(probs, 1).squeeze(-1)
                samples[:, i, j] = sampled_indices
    return samples

def train_pixelcnn(epochs:int, batchsize:int, learning_rate:float, num_workers:int=0):
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = DataSetup(batch_size=batchsize, num_workers=num_workers)
    train_dataloader, _ = data.get_train_dataloader()

    # Initialize PixelCNN
    pixel_cnn_model = PixelCNN(num_embeddings=512, embedding_dim=64, hidden_dim=128).to(device)

    # Initialize and Load PRETRAINED VQ-VAE
    vqvae_model = VQ_VAE_Model().to(device)
    checkpoint_path = "/content/vqvae/model_checkpoints/vqvae_final_trial2_codebookfix.pt"
    vqvae_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    vqvae_model.eval() # Keep VQ-VAE frozen
    for param in vqvae_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(pixel_cnn_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        pixel_cnn_model.train()
        epoch_loss = 0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")

        for images, _ in pbar:
            images = images.to(device)

            with torch.no_grad():
                # Extract target indices from frozen VQ-VAE
                _, _, _, encoding_indices = vqvae_model.encode(images)
                targets = encoding_indices.view(-1, 7, 7).long()

            optimizer.zero_grad()
            logits = pixel_cnn_model(targets)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        # --- EVALUATION AND SAMPLING ---
        print(f"Generating samples for Epoch {epoch+1}...")
        pixel_cnn_model.eval()

        # Sample new indices with temperature
        gen_indices = sample_from_pixelcnn(pixel_cnn_model, batch_size=16, device=device, temperature=1.0)

        # CHECK FOR COLLAPSE: Print unique indices used
        unique_indices = torch.unique(gen_indices)
        print(f"--- DEBUG: Unique indices in sample: {len(unique_indices)}/512 ---")

        # Decode indices back to image pixels
        with torch.no_grad():
            gen_images = vqvae_model.decode_indices(gen_indices)

        grid = make_grid(gen_images, nrow=4, normalize=True)
        save_image(grid, f"generated_epoch_{epoch+1}.png")

if __name__ == "__main__":
    # Call training with 0 workers for Colab stability
    train_pixelcnn(epochs=20, batchsize=64, learning_rate=3e-4, num_workers=0)
