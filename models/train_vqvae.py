
from data_setup import DataSetup
from models.vq_vae import VQ_VAE_Model
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from tqdm.auto import tqdm

def save_img_tensors_as_grid(img_tensors, nrows, f):
    batch_size, channels, height, width = img_tensors.shape
    ncols = batch_size // nrows

    imgs_array = img_tensors.detach().cpu().numpy()
    imgs_array = np.clip(imgs_array, 0, 1)
    imgs_array = (imgs_array * 255).astype(np.uint8)

    grid_h = nrows * height
    grid_w = ncols * width
    img_arr = np.zeros((grid_h, grid_w), dtype=np.uint8)

    for idx in range(batch_size):
        row_idx = idx // ncols
        col_idx = idx % ncols
        img_arr[row_idx*height : (row_idx+1)*height,
                col_idx*width  : (col_idx+1)*width] = imgs_array[idx, 0]

    Image.fromarray(img_arr, mode="L").save(f"{f}.jpg")

def train_vqvae(epochs:int, batchsize:int, learning_rate:float, num_workers:int):
    torch.manual_seed(42)

    data = DataSetup(batch_size=batchsize, num_workers=num_workers)
    train_dataloader, test_dataloader = data.get_train_dataloader()

    model = VQ_VAE_Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    criterion = nn.MSELoss()

    for epoch in tqdm(range(epochs)):
        model.train()
        total_train_loss = 0
        total_recon_loss = 0

        for i, (images, _) in enumerate(train_dataloader):
            optimizer.zero_grad()
            images = images.to(device)

            x_hat, dictionary_loss, commitment_loss, _ = model(images)
            recon_loss = criterion(x_hat, images)
            loss = recon_loss + dictionary_loss + 0.25*commitment_loss

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_recon_loss += recon_loss.item()

            if i % 100 == 0:
                print(f"Epoch: {epoch+1} | Batch: {i} | Train Loss: {total_train_loss/(i+1):.4f}")

        
                
        if epoch == 14:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_train_loss / len(train_dataloader),
            }
            # This line MUST be indented to match the 'checkpoint' variable above
            torch.save(checkpoint, f"/content/VQ_VAE/model_checkpoints_epoch_{epoch+1}.pth")
            print(f"Checkpoint saved at epoch {epoch+1}")

        
        model.eval()
        with torch.no_grad():
            for images, _ in test_dataloader:
                images = images.to(device)
                x_hat, _, _, _ = model(images)


                save_img_tensors_as_grid(images, 4, f"true_epoch_{epoch+1}")
                save_img_tensors_as_grid(x_hat, 4, f"recon_epoch_{epoch+1}")
                break
