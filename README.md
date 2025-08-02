## **Vision Transformer Based Autoencoder**
**Input**: 
    256×256 RGB images  
- **Latents**:
    Encoded to a 16×16×16 tensor — a 48,000 % compression
- **Dataset**:
    Trained on the MSCOCO 2017 dataset  
- **Architecture**:
    Encoder, Decoder architecture featuring a Vision Transformer, patchifying the image into 16x16 patch tokens before passing the token sequence through spatially-aware transformer blocks, embedded via RoPE
    (Rotary Position Embeddings) as per Su et al. Finally, the sequence is compressed to a 16x16x16 latent. The decoder consists of mostly the same architecture, but backwards, decompressing the 16x16x16
    latent into a sequence of tokens, passing it through spatially-aware transformer blocks, then linearly projecting the tokens back to image space.
- **Training**:
    1) VGG Perceptual Loss as per Zhang et al., 2018 [[arXiv](https://arxiv.org/abs/1801.03924)] <br>
    2) KL Divergence Loss as per Kingma et al., 2018 [[arXiv](https://arxiv.org/abs/1312.6114)]
       <br>⚠️ Note: My encoder does not output a mean and log-var (μ, logσ²) pair as in a standard VAE. Instead, it deterministically outputs a singular latent vector.  
          To encourage a Gaussian prior, I apply a quasi-KL divergence loss between the encoder output and a standard normal distribution. This acts as a regularizer, loosely encouraging the latent space to remain            centered and isotropic. <br>
    3) PatchGAN Discriminator Loss as per Isola et al., 2018 [[arXiv](https://arxiv.org/pdf/1611.07004)] <br>
