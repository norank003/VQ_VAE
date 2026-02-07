from vqvae.data_setup import DataSetup
from vqvae.models.vq_vae import VQ_VAE_Model
import torch
import tqdm.auto import tqdm

def save_img_tensors_as_grid(img_tensors, nrows, f):
    # img_tensors shape: [B, 1, 28, 28]
    batch_size, channels, height, width = img_tensors.shape
    ncols = batch_size // nrows
    
    # Move to CPU and numpy
    imgs_array = img_tensors.detach().cpu().numpy()
    
    # ADJUST THIS: If your data is 0 to 1, use this:
    imgs_array = np.clip(imgs_array, 0, 1)
    imgs_array = (imgs_array * 255).astype(np.uint8)
    
    # Create the empty grid (Height, Width) - No '3' for RGB if it's MNIST
    grid_h = nrows * height
    grid_w = ncols * width
    img_arr = np.zeros((grid_h, grid_w), dtype=np.uint8)
    
    for idx in range(batch_size):
        row_idx = idx // ncols
        col_idx = idx % ncols
        
        # Place the 28x28 patch into the big grid
        img_arr[row_idx*height : (row_idx+1)*height, 
                col_idx*width  : (col_idx+1)*width] = imgs_array[idx, 0] # [0] for greyscale channel

    # Save as "L" (Luminance/Greyscale) instead of "RGB"
    Image.fromarray(img_arr, mode="L").save(f"{f}.jpg")



def train_vqvae(self, epochs:int , batchsize:int , learning_rate:int , num_workers:int):

  torch.manual_seed(42)

  data=DataSetup(batch_size=batchsize,num_workers=num_workers)
  train_dataloader,test_dataloader=data.get_train_dataloader()

  model=VQ_VAE_Model()
  optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

  device="cuda" if torch.cuda.is_available() else "cpu"
  model.to(device)

  criterion = nn.MSELoss()

  model.train()
  for epoch in tqdm(range(epochs)):

    total_train_loss = 0
    total_recon_loss = 0
   
    for i, (images, _) in enumerate(train_dataloader)::
      optimizer.zero_grad()
      
      images = images.to(device) # toTensor in dataloader 

      x_hat, dictionary_loss, commitment_loss, encoding_indices = model(images)   #Forward pass
      recon_loss = criterion(x_hat, images)
      loss = recon_loss + dictionary_loss + commitment_loss
      loss.backward()
      optimizer.step()

      total_train_loss += loss.item()
      total_recon_loss += recon_loss.item()
      print(f"Epoch: {epoch+1}/{epochs} | Batch: {i+1}/{len(train_dataloader)} | Train Loss: {total_train_loss/(i+1):.4f} | Recon Loss: {total _recon_loss/(i+1):.4f}")

   model.eval()   


   with torch.no_grad():
      for valid_tensors in test_dataloader:
          break

          save_img_tensors_as_grid(valid_tensors[0], 4, "true")
          save_img_tensors_as_grid(model(valid_tensors[0].to(device))["x_recon"], 4, "recon")



