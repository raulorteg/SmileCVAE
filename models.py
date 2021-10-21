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
        self.latent_size = latent_size
        self.vae = vae

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels=input_size[1], out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        # Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        
        # Linear section
        out_features = 2*self.latent_size if self.vae else self.latent_size
        self.encoder_lin = nn.Sequential(
            nn.Linear(in_features=input_size[2]*input_size[3]*32, out_features=2048),
            nn.ReLU(True),
            nn.Linear(in_features=2048, out_features=out_features)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x

class Decoder(nn.Module):
    def __init__(self, output_size, latent_size, vae=False):
        super(Decoder, self).__init__()
        
        self.output_size = output_size
        self.latent_size = latent_size
        self.vae = vae

        # linear section
        #in_features = 2*self.latent_size if self.vae else self.latent_size
        in_features = self.latent_size
        self.decoder_lin = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=32*output_size[2]*output_size[3]),
            nn.ReLU()
        )

        # unflatten section
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 50, 50))

        # deconvolution section
        self.decoder_conv = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
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
        self.decode = Decoder(output_size=input_size, latent_size=latent_size+1, vae=True)
    
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