import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
from data_setup import DataSetup
from models.vq_vae import VQ_VAE_Model
from models.pixelcnn import PixelCNN

def sample_from_pixelcnn(model, batch_size, device, shape=(7, 7)):
    """Sequentially samples indices from the PixelCNN."""
    model.eval()
    samples = torch.zeros(batch_size, *shape).long().to(device)

    with torch.no_grad():
        for i in range(shape[0]):
            for j in range(shape[1]):
                logits = model(samples)
                probs = F.softmax(logits[:, :, i, j], dim=1)
                sampled_indices = torch.multinomial(probs, 1).squeeze(-1)
                samples[:, i, j] = sampled_indices

    return samples



def decode_indices_to_image(vqvae_model, indices, device):
    """Simple wrapper to call the VQ-VAE decode method"""
    vqvae_model.eval()
    with torch.no_grad():
        # This calls the method we added in step 1
        return vqvae_model.decode_indices(indices)


def train_pixelcnn(epochs:int, batchsize:int, learning_rate:float, num_workers:int):
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = DataSetup(batch_size=batchsize, num_workers=num_workers)
    train_dataloader, _ = data.get_train_dataloader()

    pixel_cnn_model = PixelCNN(num_embeddings=512, embedding_dim=64, hidden_dim=128).to(device)

    # LOAD YOUR PRETRAINED VQ-VAE HERE
    vqvae_model = VQ_VAE_Model().to(device)
    checkpoint_path = "/content/vqvae/model_checkpoints/vqvae_final.pt" # Ensure this path is correct
    state_dict = torch.load(checkpoint_path, map_location=device)
    vqvae_model.load_state_dict(state_dict)
    vqvae_model.eval()

    optimizer = torch.optim.Adam(pixel_cnn_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        pixel_cnn_model.train()
        epoch_loss = 0

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        for images, _ in pbar:
            images = images.to(device)

            with torch.no_grad():
                _, _, _, encoding_indices = vqvae_model.encode(images)
                targets = encoding_indices.view(-1, 7, 7).long()

            optimizer.zero_grad()
            logits = pixel_cnn_model(targets)

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        print(f"Generating samples for Epoch {epoch+1}...")
        pixel_cnn_model.eval()

        gen_indices = sample_from_pixelcnn(pixel_cnn_model, batch_size=16, device=device)
        gen_images = decode_indices_to_image(vqvae_model, gen_indices, device)

        grid = make_grid(gen_images, nrow=4, normalize=True)
        save_image(grid, f"generated_epoch_{epoch+1}.png")

if __name__ == "__main__":
    train_pixelcnn(epochs=10, batchsize=32, learning_rate=1e-3, num_workers=2)
