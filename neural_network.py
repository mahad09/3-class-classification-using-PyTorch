from torch.nn import Module
from torch.nn import Linear
from torch.nn import ReLU
import torch.nn.functional as F

class NeuralNetwork(Module):
    def __init__(self):
        super().__init__()
        self.layer1 = Linear(48, 50)
        self.layer2 = Linear(50, 40)
        self.layer2 = Linear(50, 40)
        self.layer3 = Linear(40, 3)
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.log_softmax(self.layer3(x), dim=1)
        
        return x

