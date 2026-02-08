import argparse
from models.train_vqvae import train_vqvae
from data_setup import DataSetup

def main():
    parser = argparse.ArgumentParser(description="Train VQ-VAE Model")

    # Define arguments
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_workers", type=int, default=2)

    args = parser.parse_args()

    # Call the function from your other file
    train_vqvae(
        epochs=args.epochs,
        batchsize=args.batch_size,
        learning_rate=args.lr,
        num_workers=args.num_workers
    )

if __name__ == "__main__":
    main()
