from torch import nn

class Generator(nn.Module):
    def __init__(self, z_dim=128, image_channels=3, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(z_dim, hidden_dim*4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim*4, hidden_dim*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim*2, hidden_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim, image_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.gen(input)