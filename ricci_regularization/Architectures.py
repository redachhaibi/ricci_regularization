import torch
import torch.nn as nn

class TorusAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(TorusAE, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        # Non-linearity
        self.non_linearity = torch.sin
        self.non_linearity2 = torch.cos # should this not be vice versa??
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc3 = nn.Linear(h_dim2, z_dim)
        # decoder part
        # Double dimension as circle is mimicked using sin and cos charts
        self.fc4 = nn.Linear(2*z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
        
    def encoder(self, x):
        h = self.non_linearity(self.fc1(x))
        h = self.non_linearity(self.fc2(h))
        h = self.fc3(h)
        # Concatenate sin and cos non-linearities
        # Warning: Done along dimension 1, as dimension 0 is the batch dimension
        #h = torch.cat( (self.non_linearity(h), self.non_linearity2(h)), 1)
        h = torch.cat( (self.non_linearity2(h), self.non_linearity(h)), 1)
        return h # Latent variable z, Wannabe uniform on the circle
    def encoder2lifting(self, x):
        h = self.non_linearity(self.fc1(x))
        h = self.non_linearity(self.fc2(h))
        h = self.fc3(h)
        # Concatenate sin and cos non-linearities
        # Warning: Done along dimension 1, as dimension 0 is the batch dimension
        #h = torch.cat( (self.non_linearity(h), self.non_linearity2(h)), 1)
        # cosphi,sinphi
        h = torch.cat( (self.non_linearity2(h), self.non_linearity(h)), 1) 
        cosphi = h[:, 0:self.z_dim]
        sinphi = h[:, self.z_dim:2*self.z_dim]
        phi = torch.acos(cosphi)*torch.sgn(torch.asin(sinphi))
        return phi
    def encoder_torus(self, x):   
        #This is a mapping to a feature space so it would be wrong to use it
        h = self.non_linearity(self.fc1(x))
        h = self.non_linearity(self.fc2(h))
        h = self.fc3(h)
        return h
        
    def decoder(self, z):
        #h = self.non_linearity( math.pi*z + self.decoderBias ) # Expects 2pi periodic non-linearity to create torus topology
        h = z
        h = self.non_linearity( self.fc4(h))
        h = self.non_linearity( self.fc5(h))
        return self.fc6(h)
        #return self.non_linearity( self.fc6(h) )
    def decoder_torus(self, z):
        h = z
        h = torch.cat( (self.non_linearity2(h), self.non_linearity(h)), -1)
        h = self.non_linearity( self.fc4(h))
        h = self.non_linearity( self.fc5(h))
        return self.fc6(h)
        #return self.non_linearity( self.fc6(h) )
    
    def forward(self, x):
        z = self.encoder(x.view(-1, self.x_dim))
        return self.decoder(z), z

class TorusConvAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim, pixels):
        super(TorusConvAE, self).__init__()
        self.pixels = pixels
        self.x_dim = x_dim
        self.z_dim = z_dim
        # Non-linearity
        self.non_linearity = torch.sin
        self.non_linearity2 = torch.cos # should this not be vice versa??
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc3 = nn.Linear(h_dim2, z_dim)
        # decoder part
        # Double dimension as circle is mimicked using sin and cos charts
        self.fc4 = nn.Linear(2*z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
        
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16, track_running_stats=False),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, self.z_dim)
        )

        self.decoder_lin = nn.Sequential(
            nn.Linear(2*self.z_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 
            stride=2, output_padding=0),
            nn.BatchNorm2d(16, track_running_stats=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, 
            padding=1, output_padding=1),
            nn.BatchNorm2d(8, track_running_stats=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, 
            padding=1, output_padding=1)
        )
        
    def encoder(self, x):
        x = x.view(-1,1,self.pixels,self.pixels)
        h = self.encoder_cnn(x)
        h = self.flatten(h)
        h = self.encoder_lin(h)
        #h = self.non_linearity(self.fc1(x))
        #h = self.non_linearity(self.fc2(h))
        #h = self.fc3(h)
        # Concatenate sin and cos non-linearities
        # Warning: Done along dimension 1, as dimension 0 is the batch dimension
        #h = torch.cat( (self.non_linearity(h), self.non_linearity2(h)), 1)
        h = torch.cat( (self.non_linearity2(h), self.non_linearity(h)), 1)
        return h # Latent variable z, Wannabe uniform on the circle
    def encoder2lifting(self, x):
        h = self.encoder(x) 
        cosphi = h[:, 0:self.z_dim]
        sinphi = h[:, self.z_dim:2*self.z_dim]
        phi = torch.acos(cosphi)*torch.sgn(torch.asin(sinphi))
        return phi
    
    def encoder_torus(self, x):   
        #This is a mapping to a feature space so it would be wrong to use it
        x = x.view(-1,1,self.pixels,self.pixels)
        h = self.encoder_cnn(x)
        h = self.flatten(h)
        h = self.encoder_lin(h)
        return h
    
    def decoder(self, z):
        #h = self.non_linearity( math.pi*z + self.decoderBias ) # Expects 2pi periodic non-linearity to create torus topology
        h = self.decoder_lin(z)
        h = self.unflatten(h)
        h = self.decoder_conv(h)
        h = h.view(-1,self.x_dim)
        return torch.sigmoid(h)
        #return self.non_linearity( self.fc6(h) )
    def decoder_torus(self, z):
        h = torch.cat( (self.non_linearity2(z), self.non_linearity(z)), 1)
        h = self.decoder(h)
        return h
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z
    