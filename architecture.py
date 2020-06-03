from torch import nn

# VARIOUS ARCHITECTURES
class CelebaCoder(nn.Module):
    def __init__(self, hidden_dim, kernel_size, kernel_num, latent_dim):
        super(CelebaCoder, self).__init__()

        self.kn = kernel_num
        kn, ks = kernel_num, kernel_size

        self.encoder_cnn_modules = [
            nn.Conv2d(3, kn, ks, stride=2, padding=ks // 2),
            nn.ReLU(),
            nn.Conv2d(kn, kn, ks, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(kn, kn * 2, ks, stride=2, padding=ks // 2),
            nn.ReLU(),
            nn.Conv2d(kn * 2, kn * 2, ks, stride=2, padding=1),
            nn.ReLU()
        ]
        self.encoder_fc_modules = [
            nn.Linear(4 * 4 * kn * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        ]

        self.encoder_cnn = nn.Sequential(*self.encoder_cnn_modules)
        self.encoder_fc = nn.Sequential(*self.encoder_fc_modules)


        self.decoder_fc_modules = [
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4 * 4 * kn * 2),
            nn.ReLU(),
        ]

        self.decoder_cnn_modules = [
            nn.ConvTranspose2d(kn * 2, kn * 2, ks, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(kn * 2, kn, ks, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(kn, kn, ks, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(kn, 3, ks, stride=2, padding=1),
            nn.Sigmoid()
        ]

        self.decoder_fc = nn.Sequential(*self.decoder_fc_modules)
        self.decoder_cnn = nn.Sequential(*self.decoder_cnn_modules)

    def preprocess(self, x):
        return x

    def encode(self, x):
        x = self.encoder_cnn(x)
        x = x.view(x.shape[0],-1)
        x = self.encoder_fc(x)
        return x

    def decode(self, x):
        x = self.decoder_fc(x)
        x = x.view(-1, self.kn * 2, 4, 4)
        x = self.decoder_cnn(x)
        return x

    def forward(self, x):
        x = self.preprocess(x)
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return encoded, decoded

class FCCoder(nn.Module):
    def __init__(self, layers_num, input_dim, hidden_dim, latent_dim):
        super(FCCoder, self).__init__()

        encoder_modules = [nn.Linear(input_dim, hidden_dim)]
        for layer_idx in range(layers_num - 2):
            encoder_modules += [nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)]
        encoder_modules += [nn.ReLU(), nn.Linear(hidden_dim, latent_dim)]
        self.encoder = nn.Sequential(*encoder_modules)

        decoder_modules = [nn.Linear(latent_dim, hidden_dim)]
        for layer_idx in range(layers_num - 2):
            decoder_modules += [nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)]
        decoder_modules += [nn.ReLU(), nn.Linear(hidden_dim, input_dim), nn.Sigmoid()]
        self.decoder = nn.Sequential(*decoder_modules)

    def preprocess(self, x):
        return x.view(x.shape[0], -1).float()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(encoded)


    def forward(self, x):
        x = self.preprocess(x)
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return encoded, decoded
