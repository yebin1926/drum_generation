# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------
# 1) SSM VAE-GAN components (Wei et al. §3.2, §4.4) - Melody -> predict drum SSM
# ----------------------------------------

class SSMEncoder(nn.Module):
    """
    Encodes a 256×256 melodic/bar-level SSM into a 32-dim latent vector.
    - 8 Conv2d layers, each stride=2, kernel=4, padding=1
    Job: squish 256x256 SSM map down into 32-number summary
    """
    def __init__(self, in_channels=1, base_channels=64, latent_dim=32):
        #base channels: how many patterns our layer will learn
        #latent_dim: how many numbers you want
        super().__init__()
        layers = [] 
        curr_ch = in_channels
        # 8 downsampling conv blocks
        for i in range(8): #8 rounds of work - at each round we will halve the input and double the # of filters
            out_ch = base_channels * min(2**i, 8)  # 64,128,256,512...
            layers.append(nn.Conv2d(curr_ch, out_ch, 4, 2, 1)) #look at each 4x4 patch of input, move it with steps of 2 pixels (so now 128x128), paste result into one of out_ch layers
            #BatchNorm & ReLU are helpers to keep things smooth & let network learn better
            layers.append(nn.BatchNorm2d(out_ch)) #normalize data to fit certain range
            layers.append(nn.ReLU(inplace=True)) #turn all neg values -> 0
            curr_ch = out_ch # our output out_ch => our next input curr_ch
        self.conv = nn.Sequential(*layers) #pack 8 rounds into 1 giant step "self.conv" and rememeber this is what the next round starts with.
        # after halving 8 times, each feature map is 1×1, now we end up with curr_ch numbers per example
        # 2) 3 fully-connected layers with skip connection
        self.fc1 = nn.Linear(curr_ch, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc_mu     = nn.Linear(512, latent_dim) #fc_mu and fc_logvar squishes that curr_ch-length vector into 32 numbers each (bottleneck part)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x):
        x = self.conv(x)              # (B, C, 1, 1)        #runs inputs through all 8 halving/filtering rounds
        x = x.flatten(1)
        # FC + skip
        h1 = F.relu(self.fc1(x))       # (B,1024)
        h2 = F.relu(self.fc2(h1) + h1) # skip connection
        h3 = F.relu(self.fc3(h2))      # (B,512)
        #latent
        mu     = self.fc_mu(h3)        # (B, latent_dim)     # mu, logvar gives two 32-number outputs, ready for the VAE latent space!!
        logvar = self.fc_logvar(h3)
        return mu, logvar


class SSMDecoder(nn.Module):
    """
    Decodes a 32-d latent to a 256×256 SSM.
    Reverse of encoder: 3 FC layers with skip, then 8 deconv upsampling.
    """
    def __init__(self, out_channels=1, base_channels=64, latent_dim=32):
        super().__init__()
        # 1) FC layers reversed
        self.fc3 = nn.Linear(latent_dim, 512)          # turns 32#s into 512#s
        self.fc2 = nn.Linear(512, 1024)                # turns 512#s into 1024#s
        self.skip_h3 = nn.Linear(512, 1024, bias=False) # << projection for skip
        self.fc1 = nn.Linear(1024, base_channels * 8)  # match encoder conv final ch
        # 2) 8 upsampling deconv layers
        deconv_layers = []                             
        curr_ch = base_channels * 8                    #now we're re-joining the 8 channels 8 times (reverse of halving)    
        for i in reversed(range(8)):
            out_ch = base_channels * min(2**(i-1), 8) if i>0 else out_channels 
            block = [nn.ConvTranspose2d(curr_ch, out_ch, kernel_size=4, stride=2, padding=1)] #grow each dim by a factor of 2, using 4x4 sliding window
            if i>0:                                                         #for first 7 rounds, 
                block += [nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]    #normalise & apply ReLU so model can learn non-linear pattern
            else:                                                           #on last round: use sigmoid to squish values into [0,1] for our SSM
                block += [nn.Sigmoid()]
            deconv_layers += block                                          #collect all 8 of these grow+normalise blocks
            curr_ch = out_ch
        self.deconv = nn.Sequential(*deconv_layers)

    def forward(self, z):
        # FC reverse stack + skip
        h3 = F.relu(self.fc3(z))                    # (B,512)        #first output of fc layer -> relu
        h2 = F.relu(self.fc2(h3) + self.skip_h3(h3))# skip           #mixes again but also adds back the old h3
        h1 = F.relu(self.fc1(h2))                   # (B, base*8)    #mixes down to final 512 numbers
        # reshape to feature map for deconv
        x = h1.view(h1.size(0), -1, 1, 1)   # (B,base*8,1,1) #jump from 512 numbers to a 512x1x1 little map
        return self.deconv(x)               # (B,1,256,256)  #let self.deconv blow it up 8 times


class SSMVAE(nn.Module):
    """Wraps SSMEncoder + reparam + SSMDecoder into VAE."""
    def __init__(self, encoder: SSMEncoder, decoder: SSMDecoder):
        super().__init__()
        self.encoder = encoder                      #the encoder class
        self.decoder = decoder                      #the decoder class

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)               #var^2 = std
        return mu + std * torch.randn_like(std)     #getting z = mu + std * N(0,1)

    def forward(self, x):
        mu, logvar = self.encoder(x)                #get mu, logvar from encoder
        z = self.reparameterize(mu, logvar)         #reparameterize for backpropagation differentiation
        recon = self.decoder(z)                     #get model's reconstruction of the input (predicted SSM)
        return recon, mu, logvar        


class SSMDiscriminator(nn.Module):
    """
    Discriminator for SSM VAE-GAN.
    Mirrors encoder conv-stack + sigmoid classifier.
    """
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()
        layers = []
        curr_ch = in_channels                       #one channel
        for i in range(8):                          
            out_ch = base_channels * min(2**i, 8)   #number of feature maps frows 64, 128, ... cap at 512
            layers += [
                nn.Conv2d(curr_ch, out_ch, kernel_size=4, stride=2, padding=1), #slide 4x4 window across input, stepping by 2 -> halves height/width from 256 till 1
                nn.LeakyReLU(0.2, inplace=True)     #like ReLU but neg values leak a bit instead of going to 0
            ]
            curr_ch = out_ch                        
        self.conv = nn.Sequential(*layers)          #stash all 8 blocks into one nn.Sequential
        self.classifier = nn.Sequential(
            nn.Flatten(),                           #turns (B, 512, 1, 1) -> (B, 512)
            nn.Linear(curr_ch, 1),                  #maps features to single logit
            nn.Sigmoid()                            #squash into [0,1] -> "prob this SSM is real"
        )

    def forward(self, x):
        x = self.conv(x)
        return self.classifier(x)


# ----------------------------------------
# 2) Drum-pattern VAE components (§3.4, §4.4)
# ----------------------------------------

class DrumEncoder(nn.Module):
    """
    Encodes an 84×96×8 bar-selection tensor into a 32-dim latent vector and note density.
    Implements: 8 conv layers ↓, 3 FC + skip, then mu/logvar and c_hat.
    """
    def __init__(self, in_channels=8, base_channels=64, latent_dim=32):
        super().__init__()
        # Conv stack
        conv_layers = []
        curr_ch = in_channels
        for i in range(8):
            out_ch = base_channels * min(2**i, 8)
            conv_layers += [
                nn.Conv2d(curr_ch, out_ch, 4, 2, 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ]
            curr_ch = out_ch
        self.conv = nn.Sequential(*conv_layers)
        # FC layers + skip
        self.fc1 = nn.Linear(curr_ch, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        # latent & density heads
        self.fc_mu     = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        self.fc_c_hat  = nn.Linear(512, 1)  # note density estimate

    def forward(self, x):
        x = self.conv(x).flatten(1)
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1) + h1)
        h3 = F.relu(self.fc3(h2))
        mu     = self.fc_mu(h3)
        logvar = self.fc_logvar(h3)
        c_hat  = self.fc_c_hat(h3)
        return mu, logvar, c_hat


class DrumDecoder(nn.Module):
    """
    Decodes (z, c_hat) into a 46×16×1 drum pattern.
    Reverse of encoder: 3 FC + skip, then 8 deconv.
    """
    def __init__(self, out_channels=1, base_channels=64, latent_dim=32):
        super().__init__()
        # input dimension = latent_dim + 1 (for c_hat)
        in_dim = latent_dim + 1
        # FC reverse
        self.fc3 = nn.Linear(in_dim, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc1 = nn.Linear(1024, base_channels * 8)
        # Deconv stack
        deconv_layers = []
        curr_ch = base_channels * 8
        for i in reversed(range(8)):
            out_ch = base_channels * min(2**(i-1), 8) if i>0 else out_channels
            block = [nn.ConvTranspose2d(curr_ch, out_ch, 4, 2, 1)]
            block += [nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)] if i>0 else [nn.Sigmoid()]
            deconv_layers += block
            curr_ch = out_ch
        self.deconv = nn.Sequential(*deconv_layers)

    def forward(self, z, c_hat):
        x = torch.cat([z, c_hat], dim=1)
        h3 = F.relu(self.fc3(x))
        h2 = F.relu(self.fc2(h3) + h3)
        h1 = F.relu(self.fc1(h2))
        x = h1.view(h1.size(0), -1, 1, 1)
        return self.deconv(x)


class DrumVAE(nn.Module):
    """Wraps DrumEncoder + reparameterize + DrumDecoder into VAE."""
    def __init__(self, encoder: DrumEncoder, decoder: DrumDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar, c_hat = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, c_hat)
        return recon, mu, logvar, c_hat