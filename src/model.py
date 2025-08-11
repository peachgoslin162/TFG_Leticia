# Código para crear el módelo de RED NEURONAL
import torch as t
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 32, dropout_prob=0.15, max_pooling=True):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()
        self.max_pooling = nn.MaxPool2d(kernel_size=2,stride=2) if max_pooling == True else nn.Identity()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        skip_connection = x
        x = self.max_pooling(x)
        return x, skip_connection

class UnsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels = 32):
        super().__init__()
        self.transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding = 1, output_padding=1)
        self.attention = AttentionGate(decoder_channels=out_channels, encoder_channels=out_channels, intermediate_channels=out_channels // 2)
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, previous_layer, skip_connection):
        x = self.transpose(previous_layer)

        #Ajusta las diferencias de tamaño entre la skip_connection y la salida de la capa transpuesta.
        diffY = skip_connection.size(2) - x.size(2)
        diffX = skip_connection.size(3) - x.size(3)

        if diffY > 0 or diffX > 0:
            skip_connection = skip_connection[:, :, :x.size(2), :x.size(3)]

        if diffY < 0 or diffX < 0:
            x = x[:, :, :skip_connection.size(2), :skip_connection.size(3)]

        x = t.cat((x, skip_connection), dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        return x

class AttentionGate(nn.Module):
    def __init__(self, decoder_channels, encoder_channels, intermediate_channels):
        super().__init__()

        # Proyección de la salida del decoder
        self.decoder_projection = nn.Sequential(
            nn.Conv2d(decoder_channels, intermediate_channels, kernel_size=1),
            nn.BatchNorm2d(intermediate_channels)
        )

        # Proyección de la skip connection del encoder
        self.encoder_projection = nn.Sequential(
            nn.Conv2d(encoder_channels, intermediate_channels, kernel_size=1),
            nn.BatchNorm2d(intermediate_channels)
        )

        # Generador del mapa de atención (resultado entre 0 y 1)
        self.attention_map = nn.Sequential(
            nn.Conv2d(intermediate_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, decoder_feature, encoder_feature):
        # Proyectamos ambas señales a un espacio común
        decoder_proj = self.decoder_projection(decoder_feature)
        encoder_proj = self.encoder_projection(encoder_feature)

        # Sumamos y aplicamos ReLU
        combined = self.relu(decoder_proj + encoder_proj)

        # Generamos el mapa de atención (valores entre 0 y 1)
        attention_weights = self.attention_map(combined)

        # Aplicamos atención: filtramos la señal del encoder
        refined_encoder = encoder_feature * attention_weights

        return refined_encoder

class AttentionUnsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels = 32):
        super().__init__()
        self.transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding = 1, output_padding=1)
        self.attention = AttentionGate(decoder_channels=out_channels, encoder_channels=out_channels, intermediate_channels=out_channels // 2)
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, previous_layer, skip_connection):
        x = self.transpose(previous_layer)

        #Ajusta las diferencias de tamaño entre la skip_connection y la salida de la capa transpuesta.
        diffY = skip_connection.size(2) - x.size(2)
        diffX = skip_connection.size(3) - x.size(3)

        if diffY > 0 or diffX > 0:
            skip_connection = skip_connection[:, :, :x.size(2), :x.size(3)]

        if diffY < 0 or diffX < 0:
            x = x[:, :, :skip_connection.size(2), :skip_connection.size(3)]

        refined_skip_connection = self.attention(x, skip_connection)

        x = t.cat((x, refined_skip_connection), dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class UNet(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 64, n_classes = 13):
        super().__init__()
        self.block1 = ConvBlock(in_channels, out_channels)
        self.block2 = ConvBlock(out_channels, out_channels * 2)
        self.block3 = ConvBlock(out_channels * 2, out_channels * 4)
        self.block4 = ConvBlock(out_channels * 4, out_channels * 8, dropout_prob=0.3)
        self.block5 = ConvBlock(out_channels * 8, out_channels * 16, dropout_prob=0.3, max_pooling=False)

        self.upblock6 = AttentionUnsamplingBlock(out_channels * 16,out_channels * 8)
        self.upblock7 = AttentionUnsamplingBlock(out_channels * 8, out_channels * 4)
        self.upblock8 = AttentionUnsamplingBlock(out_channels * 4, out_channels * 2)
        self.upblock9 = AttentionUnsamplingBlock(out_channels * 2, out_channels)

        self.final_conv = nn.Conv2d(out_channels, n_classes, kernel_size=1)

    def forward(self, x):
        x, skip1 = self.block1(x)
        x, skip2 = self.block2(x)
        x, skip3 = self.block3(x)
        x, skip4 = self.block4(x)
        x, _ = self.block5(x)

        x = self.upblock6(x, skip4)
        x = self.upblock7(x, skip3)
        x = self.upblock8(x, skip2)
        x = self.upblock9(x, skip1)

        output = self.final_conv(x)

        return output
