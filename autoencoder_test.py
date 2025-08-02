from autoencoder import Encoder, Decoder
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

image_shape = 256
emb_dim = 768
patch_size = 16

encoder = Encoder(latent_dim=16, 
                  image_shape=image_shape, 
                  emb_dim=emb_dim, 
                  patch_size=patch_size)
encoder.load_state_dict(torch.load("encoder16.pt", map_location=torch.device('cpu')))

decoder = Decoder(latent_dim=16,
                  image_shape=image_shape, 
                  emb_dim=emb_dim, 
                  patch_size=patch_size)
decoder.load_state_dict(torch.load("decoder16.pt", map_location=torch.device('cpu')))

image = cv2.imread("test_image.jpg")
image = cv2.resize(image, (image_shape, image_shape))
image = torch.tensor(image, dtype=torch.float32, device='cpu').permute(2, 0, 1) / 127.5 - 1.0
image = image.unsqueeze(0) 
with torch.no_grad():
    z = encoder(image)
    x = decoder(z)
plt.imshow(x[0].permute(1, 2, 0).numpy()*0.5 + 0.5)
plt.show()