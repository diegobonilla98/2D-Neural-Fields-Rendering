import torch
from torch import nn
from torch.autograd import Variable


class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class NeuralRendererModel2D(nn.Module):
    def __init__(self, w0=1.):
        super(NeuralRendererModel2D, self).__init__()

        self.fpn = nn.Sequential(
            nn.Linear(2, 256),
            Sine(w0),

            *[nn.Sequential(
                nn.Linear(256, 256),
                Sine(w0)
            ) for _ in range(5)],

            nn.Linear(256, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fpn(x)


if __name__ == '__main__':
    model = NeuralRendererModel2D()
    print(model)
    model = model.cuda()
    model.eval()
    image = torch.rand((1, 1, 2)).cuda()
    image = Variable(image)
    out = model(image)
    print(out)
