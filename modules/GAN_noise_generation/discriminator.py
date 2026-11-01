import torch.nn as nn
import torch

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class CIFARTruckDiscriminatorFixed(nn.Module):
    def __init__(self, in_channels=3, features=[32, 64, 128]):
        super().__init__()
        
        # Entrée : 32x32 (6 canaux suite à la concaténation)
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels * 2, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"), # Sortie : 16x16
            nn.LeakyReLU(0.2, inplace=True)
        )

        layers = []
        in_channels = features[0]
        
        # Bloc 1 : 16x16 -> stride 2 -> 8x8
        layers.append(CNNBlock(in_channels, features[1], stride=2))
        
        # Bloc 2 : 8x8 -> stride 2 -> 4x4
        layers.append(CNNBlock(features[1], features[2], stride=2))

        self.model = nn.Sequential(*layers)
        
        # COUCHE FINALE CORRIGÉE :
        # On utilise un kernel_size=1 pour préserver la dimension 4x4 acquise au bloc précédent.
        self.final = nn.Conv2d(features[2], 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x, y):
        # Concaténation de l'image conditionnelle et du camion
        torch_cat = torch.cat([x, y], dim=1)
        
        out = self.initial(torch_cat)
        out = self.model(out)
        return self.final(out) # Renvoyait 1x1x1, renvoie maintenant 1x4x4 !
