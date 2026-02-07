from vqvae.data_setup import DataSetup
from vqvae.models.vq_vae import VQ_VAE_Model
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

def save_img_tensors_as_grid(img_tensors, nrows, f):
    img_tensors = img_tensors.permute(0, 2, 3, 1)
    imgs_array = img_tensors.detach().cpu().numpy()
    imgs_array[imgs_array < -0.5] = -0.5
    imgs_array[imgs_array > 0.5] = 0.5
    imgs_array = 255 * (imgs_array + 0.5)
    batch_size = img_tensors.shape[0]
    img_size = img_tensors.shape[1]
    ncols = batch_size // nrows
    img_arr = np.zeros((nrows * img_size, ncols * img_size, 3))
    for idx in range(batch_size):
        row_idx = idx // ncols
        col_idx = idx % ncols
        row_start = row_idx * img_size
        row_end = row_start + img_size
        col_start = col_idx * img_size
        col_end = col_start + img_size
        img_arr[row_start:row_end, col_start:col_end] = imgs_array[idx]

    Image.fromarray(img_arr.astype(np.uint8), "RGB").save(f"{f}.jpg")

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

            x_hat, dictionary_loss, commitment_loss, encoding_indices = model(images)
            recon_loss = criterion(x_hat, images)
            loss = recon_loss + dictionary_loss + commitment_loss
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_recon_loss += recon_loss.item()
            print(f"Epoch: {epoch+1}/{epochs} | Batch: {i+1}/{len(train_dataloader)} | Train Loss: {total_train_loss/(i+1):.4f} | Recon Loss: {total_recon_loss/(i+1):.4f}")

        model.eval()   
        with torch.no_grad():
            for valid_tensors, _ in test_dataloader:
                save_img_tensors_as_grid(valid_tensors, 4, f"true_epoch_{epoch}")
                # Accessing index 0 because model returns (x_recon, d_loss, c_loss, indices)
                recon_out = model(valid_tensors.to(device))[0]
                save_img_tensors_as_grid(recon_out, 4, f"recon_epoch_{epoch}")
                break
