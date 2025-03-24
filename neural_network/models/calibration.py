import torch
import torch.nn as nn
import torch.nn.functional as F

class ParameterCalibrationModule(nn.Module):
    def __init__(self, hidden1, hidden2):
        super(ParameterCalibrationModule, self).__init__()
        self.fc1 = nn.Linear(3, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 3)
        
    def forward(self, p):
        """
        输入: p = [θₐ, d, θₚ] (垂直孔径, 海床深度, 声纳俯角)
        输出: p' = [θₐ', d', θₚ'] (校准后的参数)
        """
        h1 = F.relu(self.fc1(p))
        h2 = F.relu(self.fc2(h1))
        p_calib = self.fc3(h2)
        return p_calib