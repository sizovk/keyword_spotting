import torch
import torchaudio
from torch import nn


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


class CRNNStream(nn.Module):

    def __init__(
        self,
        cnn_out_channels,
        kernel_size,
        stride,
        n_mels,
        gru_num_layers,
        bidirectional,
        hidden_size,
        num_classes,
        max_window_length, 
        streaming_step_size
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
        
        self.gru = nn.RNN(
            input_size=self.conv_out_frequency * cnn_out_channels,
            hidden_size=hidden_size,
            num_layers=gru_num_layers,
            dropout=0.1,
            bidirectional=bidirectional,
            batch_first=True
        )

        self.attention = Attention(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)

        self.max_window_length = max_window_length
        self.streaming_step_size = streaming_step_size
        self.cache = torch.Tensor([])
        self.last_hidden = torch.Tensor([])
        self.last_prob = torch.tensor(0.0)
        self.ongoing_frames = 0
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=40
        )
    
    def forward(self, input):
        input = input.unsqueeze(dim=1)
        conv_output = self.conv(input).transpose(-1, -2)
        gru_output, output_hidden = self.gru(conv_output)
        contex_vector = self.attention(gru_output)
        output = self.classifier(contex_vector)
        return output
    
    def forward_stream(self, input, hidden):
        input = input.unsqueeze(dim=1)
        conv_output = self.conv(input).transpose(-1, -2)
        if len(hidden) > 0:
            gru_output, output_hidden = self.gru(conv_output, hidden)
        else:
            gru_output, output_hidden = self.gru(conv_output)
        contex_vector = self.attention(gru_output)
        output = self.classifier(contex_vector)
        return output, output_hidden
    
    @torch.jit.export
    def forward_chunk(self, chunk):
        self.cache = torch.concat((self.cache, chunk))
        if len(self.cache) > self.max_window_length:
            self.cache = self.cache[-self.max_window_length:]
        self.ongoing_frames += len(chunk)
        if self.ongoing_frames >= self.streaming_step_size:
            self.ongoing_frames = 0
            melspec = torch.log(self.melspec(self.cache.unsqueeze(dim=0)).clamp_(min=1e-9, max=1e9))
            output, output_hidden = self.forward_stream(melspec, hidden=self.last_hidden)
            self.last_hidden = output_hidden
            self.last_prob = torch.nn.functional.softmax(output.squeeze(dim=0))[1]
            
        return self.last_prob