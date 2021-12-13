import torch
import torch.nn as nn

class ParamGaussian(torch.distributions.Distribution):
    def __init__(self, mu, log_sigma):
        self.mu = mu
        self.sigma = log_sigma.exp()
    
    def get_eps(self):
        return torch.rand_like(self.sigma)
    
    def rsample(self):
        eps = self.get_eps()
        return self.mu + self.sigma*eps
    
    def log_prob(self, z):
        normal_dist = torch.distributions.Distribution.normal.Normal(loc=self.mu, scale=self.sigma, validate_args=False)
        return normal_dist.log_prob(z)

class Encoder(nn.Module):
    def __init__(self, input_size, latent_size, vae=False):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.input_channels = input_size[1]
        self.input_height = input_size[2]
        self.input_width = input_size[3]

        self.latent_size = latent_size
        self.vae = vae

        self.encoder_cnn = nn.Sequential(
            # e.g [3,50,50]
            nn.Conv2d(in_channels=self.input_channels, out_channels=32, kernel_size=6, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # e.g [32,25,25]
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            # e.g [64,13,13]
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            # e.g [128,7,7]
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            # e.g [256,4,4]
        )
        
        # Flatten layer  e.g [256,4,4] -> 256x4x4=4096
        self.flatten = nn.Flatten(start_dim=1)
        
        # Linear section  e.g 4096 -> 2*latent_space
        latent_size = 2*self.latent_size if self.vae else self.latent_size
        self.encoder_lin = nn.Sequential(
            nn.Linear(in_features=4096, out_features=1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=latent_size)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x) # [3,50,50] -> [256,4,4]
        x = self.flatten(x)     # [256,4,4] -> 4096
        x = self.encoder_lin(x) # 4096 -> 2*latent_size
        return x

class Decoder(nn.Module):
    def __init__(self, output_size, latent_size):
        super(Decoder, self).__init__()
        
        self.output_size = output_size
        self.out_channels = output_size[1]
        self.out_height = output_size[2]
        self.out_width = output_size[3]
        self.latent_size = latent_size

        # linear section (e.g latent_size -> 4096)
        self.decoder_lin = nn.Sequential(
            nn.Linear(in_features=latent_size, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=4096),
            nn.LeakyReLU(),
        )

        # unflatten section (e.g 4096 -> [256,4,4])
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(256, 4, 4))

        # deconvolution section ([256,4,4] -> [3,50,50])
        self.decoder_conv = nn.Sequential(
            # [256,4,4] -> [128,7,7]
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            # [128,7,7] -> [64,13,13]
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            # [64,13,13] -> [32,25,25]
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # [32,25,25] -> [3,50,50]
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=6, stride=2, padding=2),
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)  # latent_size -> 4096
        x = self.unflatten(x)    # 4096 -> [256,4,4]
        x = self.decoder_conv(x) # [256,4,4] -> [3,50,50]
        x = torch.sigmoid(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self, input_shape, latent_size):
        super(Autoencoder, self).__init__()
        self.input_shape = input_shape
        self.latent_size = latent_size

        self.encode = Encoder(input_size=input_shape, latent_size=latent_size)
        self.decode = Decoder(output_size=input_shape, latent_size=latent_size)
    
    def forward(self, x):
        z = self.encode(x)
        xhat = self.decode(z)
        return xhat

class VAE(nn.Module):
    def __init__(self, input_size, latent_size):
        super(VAE, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size

        self.register_buffer("prior_params", torch.zeros(torch.Size([1, 2*self.latent_size])))

        self.encode = Encoder(input_size=input_size, latent_size=latent_size, vae=True)
        self.decode = Decoder(output_size=input_size, latent_size=latent_size, vae=True)
    
    def posterior(self, input_):
        h = self.encode(input_)
        mu, log_sigma = torch.chunk(h, chunks=2, dim=-1)
        return ParamGaussian(mu=mu, log_sigma=log_sigma)
    
    def prior(self, batch_size):
        prior_params = self.prior_params.expand(batch_size, *self.prior_params.shape[-1:])
        mu, log_sigma = torch.chunk(prior_params, chunks=2, dim=-1)
        return ParamGaussian(mu=mu, log_sigma=log_sigma)
    
    def observation_model(self, z):
        xhat = self.decode(z)
        return xhat
    
    def sample_prior(self, batch_size):
        latent_dist = self.prior(batch_size=batch_size)
        z = latent_dist.rsample()
        return self.observation_model(z), z

    def forward(self, x):
        batch_size = x.shape[0]
        qz = self.posterior(input_=x)
        pz = self.prior(batch_size=batch_size)
        z = qz.rsample()
        xhat = self.observation_model(z)
        return {"pz":pz, "qz":qz, "z":z, "x":x, "xhat":xhat}


class CVAE(nn.Module):
    def __init__(self, input_size, latent_size):
        super(CVAE, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size

        self.register_buffer("prior_params", torch.zeros(torch.Size([1, 2*self.latent_size])))

        self.encode = Encoder(input_size=input_size, latent_size=latent_size, vae=True)
        self.decode = Decoder(output_size=input_size, latent_size=latent_size+1)
    
    def posterior(self, input_):
        h = self.encode(input_)
        mu, log_sigma = torch.chunk(h, chunks=2, dim=-1)
        return ParamGaussian(mu=mu, log_sigma=log_sigma)
    
    def prior(self, batch_size):
        prior_params = self.prior_params.expand(batch_size, *self.prior_params.shape[-1:])
        mu, log_sigma = torch.chunk(prior_params, chunks=2, dim=-1)
        return ParamGaussian(mu=mu, log_sigma=log_sigma)
    
    def observation_model(self, z):
        xhat = self.decode(z)
        return xhat
    
    def sample_prior(self, batch_size, c):
        latent_dist = self.prior(batch_size=batch_size)
        z = latent_dist.rsample()
        if z.is_cuda:
            c = c*torch.unsqueeze(torch.ones((batch_size)),1).cuda()
        else:
            c = c*torch.unsqueeze(torch.ones((batch_size)),1)
        z_c = torch.cat((z,c), dim=1)
        return self.observation_model(z_c), z_c

    def forward(self, x, c):
        batch_size = x.shape[0]
        qz = self.posterior(input_=x)
        pz = self.prior(batch_size=batch_size)
        z = qz.rsample()
        c = torch.unsqueeze(c,dim=1)
        z_c = torch.cat((z, c), dim=1).float()
        xhat = self.observation_model(z_c)
        return {"pz":pz, "qz":qz, "z":z, "z_c":z_c, "x":x, "xhat":xhat}