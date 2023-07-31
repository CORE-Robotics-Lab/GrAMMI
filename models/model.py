

import torch.nn as nn
class Model(nn.Module):
    def __init__(self, encoder, decoder):
        super(Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, obs):
        # obs = obs.to(self.device).float()
        if self.encoder is None:
            y = self.decoder(obs)
        else:
            z = self.encoder(obs)
            y = self.decoder(z)
        return y

    def predict(self, obs):
        if self.encoder is None:
            y = self.decoder(obs)
        else:
            z = self.encoder(obs)
            y = self.decoder(z)
        return y

    def compute_loss(self, obs, target):
        # obs = obs.to(self.device).float()
        target = target.to(self.device).float()
        if self.training:
            out = self.forward(obs)
        else:
            out = self.predict(obs)
        return self.decoder.compute_loss(out, target)

    def get_stats(self, obs, target):
        target = target.to(self.device).float()
        if self.training:
            out = self.forward(obs)
        else:
            out = self.predict(obs)
        return self.decoder.get_stats(out, target)

    @property
    def device(self):
        return next(self.parameters()).device