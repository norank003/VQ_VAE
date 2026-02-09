# VQVAE

To run VQVAE train write   


!python vqvae/main.py --epochs 5 --batch_size 32 --lr 0.0002 --num_workers 2


to train pixel cnn 

from vqvae.train_pixelcnn import train_pixelcnn

train_pixelcnn(
    epochs=5,
    batchsize=64,
    learning_rate=3e-4,
    num_workers=0
)
