import torch
import torch.nn.functional as F
from IPython.display import clear_output

# @title Trainer
class Trainer():
    def __init__(self, encoder, decoder, D, vgg, losses, data_len, ema=3, a_disc=1, a_vae=1, a_KL=0.1, isViT=True):
        self.vgg_schedule = None
        self.ema = 2/(ema+1)
        self.a_disc = a_disc
        self.a_vae = a_vae
        self.a_KL = a_KL

        self.isViT = isViT
        self.encoder = encoder
        self.decoder = decoder
        self.D = D
        self.vgg = vgg
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(),  lr=1e-5)
        self.encoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.encoder_optimizer, T_max=50)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(),  lr=1e-5)
        self.decoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.decoder_optimizer, T_max=50)
        self.D_optimizer = torch.optim.Adam(self.D.parameters(),  lr=4e-5)
        self.D_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.D_optimizer, T_max=50)
        self.losses = losses
        self.loss_vals = {loss:0 for loss in losses}
        self.data_len = data_len
        self.loss_record = []
        self.epoch = 1
        self.index = 1
        self.device = torch.device("cuda")

        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.D.to(self.device)
        self.vgg.to(self.device)

    def train_step(self, x, with_mse=False, freeze_ae=False, freeze_disc=False):
        self.index += 1
        x = x.to(self.device)
        with torch.no_grad():
            x_hat = self.decoder(self.encoder(x.permute(0,2,3,1))).permute(0,3,1,2) if not self.isViT else self.decoder(self.encoder(x))
        if not freeze_disc:
            disc_loss = F.relu(1. - self.D(x)).mean() + F.relu(1. + self.D(x_hat)).mean() # Hinge
            self.D_optimizer.zero_grad()
            disc_loss.backward()
            self.D_optimizer.step()
            self.D_scheduler.step()

        if not freeze_ae:
            z = self.encoder(x.permute(0,2,3,1)) if not self.isViT else self.encoder(x)
            x_hat = self.decoder(z).permute(0,3,1,2) if not self.isViT else self.decoder(z)
            mse = F.mse_loss(x_hat, x)
            KL = 0.5 * (z.mean() ** 2)
            vgg_real = self.vgg(x)
            vgg_fake = self.vgg(x_hat)
            vgg_loss = 0
            for i in range(len(vgg_real)):
                vgg_loss += F.mse_loss(vgg_real[i], vgg_fake[i]) 

            adv_loss = 0
            if not freeze_disc:
                adv_loss = -(self.D(self.decoder(self.encoder(x))).mean())

            loss = mse * with_mse + self.a_KL* KL + vgg_loss + self.a_vae * adv_loss
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            loss.backward()
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
            self.encoder_scheduler.step()
            self.decoder_scheduler.step()

        self.update_batch({"mse":mse.item() if not freeze_ae else 0,
                           "gan":disc_loss.item() if not freeze_disc else 0,
                           "vgg":vgg_loss.item() if not freeze_ae else 0,
                           "KL":z.mean() if not freeze_ae else 0})

    def update_batch(self, loss_vals):
        clear_output(wait=True)
        for record in self.loss_record:
            print(record)
        self.loss_vals = {loss:(1-self.ema)*self.loss_vals[loss] + self.ema*loss_vals[loss] for loss in self.losses}
        print(f"epoch:{self.epoch} ", end="")
        for loss in self.losses:
            print(f"{loss}: {self.loss_vals[loss]:.3f} ", end="")
        for _ in range(int(self.index * 20 / self.data_len)):
            print("=", end="")
        for _ in range(int(self.index * 20 / self.data_len),20):
            print("-", end="")

    def update_epoch(self):
        self.index = 0
        record = f"epoch:{self.epoch} "
        for loss in self.losses:
            record += f"{loss}: {self.loss_vals[loss]:.3f} "
        self.loss_record.append(record)
        self.epoch += 1