import kagglehub
import cv2
import os
from IPython.display import clear_output
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from autoencoder import Encoder, Decoder
from trainer import Trainer
from objectives import Discriminator, vgg_builder

# Global Parameters
image_shape = 256
emb_dim = 768
patch_size = 16

image_path = kagglehub.dataset_download("awsaf49/coco-2017-dataset")
data = []
for dirpath, _, filenames in os.walk(image_path):
    for filename in filenames:
        if filename.endswith("jpg"):
            name = os.path.join(dirpath, filename)
            img = cv2.imread(name)
            img = cv2.resize(img, (image_shape,image_shape))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 127.5 - 1.0
            img = torch.tensor(img).permute(2,0,1)
            data.append(img)
            clear_output(wait=1)
            print(f"{len(data)/1670:.2f}%")
print(len(data))

class CustomDataset(Dataset):
    def __init__(self, data):
        self.indices = np.arange(len(data))
        np.random.shuffle(self.indices)
        self.data = data

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return torch.tensor(self.data[self.indices[idx]], dtype=torch.float32)

# Sanity Check :)
plt.imshow(CustomDataset(data)[0].permute(1,2,0)/2+0.5)

encoder = Encoder(latent_dim=16)
decoder = Decoder(latent_dim=16)
D = Discriminator((3,256,256))

vgg = vgg_builder()
for param in vgg.parameters():
  param.requires_grad = False
vgg.eval()

print(f"encoder: {sum(p.numel() for p in encoder.parameters())/(262144):.3f}MB")
print(f"decoder: {sum(p.numel() for p in decoder.parameters())/(262144):.3f}MB")
print(f"Discriminator: {sum(p.numel() for p in D.parameters())/(262144):.3f}MB")
print(f"VGG: {sum(p.numel() for p in vgg.parameters())/(262144):.3f}MB")

batch_size = 16
dataset    = CustomDataset(data)
loader     = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=8,
                        pin_memory=True)
epochs = 5
trainer = Trainer(encoder, decoder, D, vgg, ["mse", "gan", "vgg", "KL"], len(loader) if "loader" in locals() else 0, isViT=1)
for epoch in range(1, epochs):
    index = 0
    for i, x in enumerate(loader):
        trainer.train_step(x, freeze_disc=0, with_mse=1, freeze_ae=0)
    trainer.update_epoch()

torch.save(encoder.state_dict(), "encoder16.pt")
torch.save(decoder.state_dict(), "decoder16.pt")
torch.save(D.state_dict(), "discriminator16.pt")