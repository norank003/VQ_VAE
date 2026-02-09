import torch.nn as nn
import torch.nn.functional as F
import torch





class Residual_Block(nn.Module):  # RELU 3X3CONV RELU 1x1Conv
  def __init__(self,dim):
    super().__init__()
    self.block=nn.Sequential(
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=3,stride=1,padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=1,stride=1),


    )
  def forward(self,x):
    return x+self.block(x)


class Encoder_Block(nn.Module):
  def __init__(self, input_shape: int, hidden_dims=256):
    super().__init__()
    self.encoder=nn.Sequential(
        nn.Conv2d(in_channels=input_shape,out_channels=hidden_dims//2,kernel_size=4,stride=2,padding=1), #(1,28,28)
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=hidden_dims//2  ,  out_channels=hidden_dims,  kernel_size=4  ,  stride=2  ,  padding=1), #(256,14,14)

        Residual_Block(hidden_dims),
        Residual_Block(hidden_dims),

        nn.Conv2d(hidden_dims, hidden_dims, kernel_size=3, padding=1)


    )

  def forward(self,x) :
    return self.encoder(x)




class Vector_Quantizer_Block (nn.Module):
  def __init__(self, num_embeddings=512, embedding_dim=256, commitment_cost=0.25):
    super().__init__()


    self.emmbedings=num_embeddings  # k
    self.embedding_dim=embedding_dim  #d in paper
    self.commitment_cost=commitment_cost

    #codebook creation  #Deepmind prevent exploding or encoding in small area of codebook Next parmtrize
    limit = 3 ** 0.5
    e_i_ts = torch.FloatTensor(embedding_dim, num_embeddings).uniform_(-limit, limit)
    self.register_parameter("e_i_ts", nn.Parameter(e_i_ts))

  def  forward (self,x):

    #make encoder shape into B W H C
    Permuted_x=x.permute(0,2,3,1)
    #flatten
    flatten_x=Permuted_x.contiguous().view(-1,self.embedding_dim)
    distances = (
            (flatten_x ** 2).sum(1, keepdim=True)     # Eculiden distance between flatten x my encoder output vector and codebook e_i_ts
            - 2 * flatten_x @ self.e_i_ts
            + (self.e_i_ts ** 2).sum(0, keepdim=True)
        )

    encoding_indices = distances.argmin(1) # indcies matches in codebook

    quantized_x = F.embedding(   # indcies are mapped to vectors of codebook
            encoding_indices.view(x.shape[0], *x.shape[2:]), self.e_i_ts.transpose(0, 1)
        ).permute(0, 3, 1, 2)

    dictionary_loss = ((x.detach() - quantized_x) ** 2).mean()   #2nd
    commitment_loss = ((x - quantized_x.detach()) ** 2).mean()    #3rd

    #Backpropagation pypass the vector quantize
    quantized_x = x + (quantized_x - x).detach()





    return  (
            quantized_x,
            dictionary_loss,
            commitment_loss,
            encoding_indices.view(x.shape[0], -1),
    )







class Decoder_Block (nn.Module):
  def __init__(self,hidden_dims,out_channels):
    super().__init__()

    self.decoder=nn.Sequential(
        Residual_Block(hidden_dims),
        Residual_Block(hidden_dims),
        nn.ConvTranspose2d(hidden_dims,hidden_dims,4,stride=2,padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(hidden_dims,out_channels,4,stride=2,padding=1),
        nn.Sigmoid()
    )
  def forward(self,x) :

    return self.decoder(x)






class VQ_VAE_Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder=Encoder_Block(input_shape=1,hidden_dims=256)
    self.vq_layer=Vector_Quantizer_Block(num_embeddings=512, embedding_dim=256)
    self.decoder=Decoder_Block(hidden_dims=256,out_channels=1)
  def encode(self, x):
        """Used during PixelCNN Training to get target indices"""
        z = self.encoder(x)
        # returns (quantized_x, dict_loss, commit_loss, indices)
        return self.vq_layer(z) 

 
  def decode_indices(self, indices):
        """Converts sampled indices (B, 7, 7) into an image (B, 1, 28, 28)"""
        # 1. Map indices to codebook vectors: (B, 7, 7) -> (B, 7, 7, 256)
        # We use vq_layer.e_i_ts because that is your codebook parameter
        z_q = F.embedding(indices, self.vq_layer.e_i_ts.transpose(0, 1))
        
        # 2. Permute to (B, 256, 7, 7) for the decoder
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        # 3. Pass through the decoder
        return self.decoder(z_q)
 

  def forward(self,x):
    z=self.encoder(x)
    z,dictionary_loss,commitment_loss,encoding_indices=self.vq_layer(z)
    x_hat=self.decoder(z)
    return (x_hat,dictionary_loss,commitment_loss,encoding_indices)


