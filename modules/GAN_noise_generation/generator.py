import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Convolution -> BatchNorm -> ReLU) x 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNetCIFARGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[32, 64, 128],dropout_p=0.5):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Ajout de Dropout pour injecter du bruit/stochasticité dans le décodeur
        self.dropout = nn.Dropout(p=dropout_p)

        # --- ENCODEUR ---
        # 32x32 -> 16x16 -> 8x8
        curr_in_channels = in_channels
        for feature in features:
            self.downs.append(DoubleConv(curr_in_channels, feature))
            curr_in_channels = feature

        # --- GOULOT D'ÉTRANGLEMENT (Bottleneck) ---
        # L'image arrive ici en taille 4x4 si features a 3 niveaux
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # --- DÉCODEUR ---
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # --- COUCHE FINALE ---
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.tanh = nn.Tanh() 

    def forward(self, x):
        skip_connections = []

        # Passage dans l'encodeur
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Goulot d'étranglement
        x = self.bottleneck(x)
        
        # Inversion des connexions de saut
        skip_connections = skip_connections[::-1]

        # Passage dans le décodeur
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x) # Upsample
            skip_connection = skip_connections[idx // 2]
            
            # Concaténation
            concat_x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_x) # DoubleConv
            
            # Application du Dropout sur les premières couches du décodeur pour simuler le bruit
            if idx // 2 < 2: 
                x = self.dropout(x)

        return self.tanh(self.final_conv(x))