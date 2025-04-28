import torch
import torch.nn as nn

class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))  # keep dim 0, 2, 3! only 1 is being noised
        
    def forward(self, x):
        batch, _, height, width = x.shape
        noise = torch.randn(batch, 1, height, width, device = x.device)
        return x + self.weight * noise

    

class Generator (nn.Module):
    def __init__(self, latent_dim, style_dim, features_d):
        super(Generator, self).__init__()
        
        self.features_d = features_d
        self.latent_dim = latent_dim
        self.style_dim = style_dim
        self.channels_for_convolution = int(latent_dim/16)
        
        def ffn_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.LayerNorm(out_channels),
                nn.LeakyReLU(0.01, inplace=True)
            )
        
        
        def deconvolution_block(in_channels, out_channels, kernel_size, stride, padding = 1):
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
                nn.BatchNorm2d(out_channels),                # normalizing the whole batch 
                NoiseInjection(out_channels),                # every block contains Nois-Injection in Dim: 1
                nn.LeakyReLU(0.01, inplace = True)
            )
        
        
        self.ffn = nn.Sequential(
            ffn_block(self.latent_dim, self.features_d * 25),
            ffn_block(self.features_d * 25, self.features_d * 50),
            ffn_block(self.features_d * 50, self.features_d * 50),
            ffn_block(self.features_d * 50, self.features_d * 25),
            ffn_block(self.features_d * 25, self.latent_dim)
        )
        
        
        self.dcv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),                                 
            deconvolution_block(self.channels_for_convolution, features_d * 60, kernel_size=3, stride=1),       
            deconvolution_block(self.features_d * 60, features_d * 60, kernel_size=3, stride=1),
            deconvolution_block(self.features_d * 60, features_d * 60, kernel_size=3, stride=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),                               
            deconvolution_block(self.features_d * 60, features_d * 60, kernel_size=3, stride=1),
            deconvolution_block(self.features_d * 60, features_d * 60, kernel_size=3, stride=1),
            deconvolution_block(self.features_d * 60, features_d * 60, kernel_size=3, stride=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),                          
            deconvolution_block(self.features_d * 60, features_d * 60, kernel_size=3, stride=1),
            deconvolution_block(self.features_d * 60, features_d * 60, kernel_size=3, stride=1),
            deconvolution_block(self.features_d * 60, features_d * 60, kernel_size=3, stride=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),                           
            deconvolution_block(self.features_d * 60, features_d * 60, kernel_size=3, stride=1),
            deconvolution_block(self.features_d * 60, features_d * 60, kernel_size=3, stride=1),
            deconvolution_block(self.features_d * 60, features_d * 60, kernel_size=3, stride=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),                          
            deconvolution_block(self.features_d * 60, features_d * 60, kernel_size=3, stride=1),
            deconvolution_block(self.features_d * 60, features_d * 60, kernel_size=3, stride=1),
            deconvolution_block(self.features_d * 60, features_d * 30, kernel_size=3, stride=1),
            nn.Conv2d(self.features_d * 30, 3, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid()
            
        )
        
    
    def forward(self, x, batch_size):
        
        for layer in self.ffn:
            x = layer(x)
        
        # Shape ändern, damit Convolution-Layers angewandt werden können
        x = x.view(batch_size, self.channels_for_convolution, 4, 4)
        

        for layer in self.dcv:
            x = layer(x)
            
        
        return x
