import torch
import torch.nn as nn

class Discriminator (nn.Module):
    def __init__(self, img_channels, features_d):
        super(Discriminator, self).__init__()
        
        self.features_d = features_d
        self.img_channels = img_channels
        
        def conv_block(in_channels, out_channels, kernel_size, stride, padding = 1):
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Dropout2d(0.4) 
            )
        
        def ffn_Block(in_channels, out_channels):
            return nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.LayerNorm(out_channels),
                nn.LeakyReLU(0.01, inplace=True)
            )
        
        self.disc = nn.Sequential(
            conv_block(self.img_channels, self.features_d * 40, kernel_size=3, stride=2),
            conv_block(self.features_d * 40, self.features_d * 40, kernel_size=3, stride=1),
            conv_block(self.features_d * 40, self.features_d * 40, kernel_size=3, stride=2),
            conv_block(self.features_d * 40, self.features_d * 40, kernel_size=3, stride=1),
            conv_block(self.features_d * 40, self.features_d * 40, kernel_size=3, stride=2),
            conv_block(self.features_d * 40, self.features_d * 40, kernel_size=3, stride=1),
            conv_block(self.features_d * 40, self.features_d * 40, kernel_size=3, stride=2),
            conv_block(self.features_d * 40, self.features_d * 20, kernel_size=3, stride=1),
            
        )
        
        self.ffn = nn.Sequential(
            ffn_Block(3840, features_d * 10),
            ffn_Block(features_d * 10, features_d * 100),
            nn.Linear(features_d * 100, 1),
            #nn.Sigmoid()                # Deaktiviert, solange mixed precision training aktiv (nummerische instabilität)
        )
        
        
        
    def forward(self, x):
        for layer in self.disc:
            x = layer(x)

        # Flatten für die Denseschicht
        x = torch.flatten(x, start_dim=1)

          
        for layer in self.ffn:
            x = layer(x)


        return x
