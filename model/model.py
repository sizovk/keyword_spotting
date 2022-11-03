import torch
from torch import nn

from base import BaseModel


class Attention(nn.Module):

    def __init__(self, hidden_size: int):
        super().__init__()

        self.energy = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, input):
        energy = self.energy(input)
        alpha = torch.softmax(energy, dim=-2)
        return (input * alpha).sum(dim=-2)


class CRNN(BaseModel):

    def __init__(
        self,
        cnn_out_channels,
        kernel_size,
        stride,
        n_mels,
        gru_num_layers,
        bidirectional,
        hidden_size,
        num_classes
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=cnn_out_channels,
                kernel_size=kernel_size, stride=stride
            ),
            nn.Flatten(start_dim=1, end_dim=2),
        )

        self.conv_out_frequency = (n_mels - kernel_size[0]) // stride[0] + 1
        
        self.gru = nn.GRU(
            input_size=self.conv_out_frequency * cnn_out_channels,
            hidden_size=hidden_size,
            num_layers=gru_num_layers,
            dropout=0.1,
            bidirectional=bidirectional,
            batch_first=True
        )

        self.attention = Attention(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, input):
        input = input.unsqueeze(dim=1)
        conv_output = self.conv(input).transpose(-1, -2)
        gru_output, _ = self.gru(conv_output)
        contex_vector = self.attention(gru_output)
        output = self.classifier(contex_vector)
        return output