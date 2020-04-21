import torch 
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, input_size=300, seq_size=20, output_size=512, p=0.3):
        super(LeNet, self).__init__()
        self.output_size = output_size
        self.kernel_size = 5
        self.fc_shape1 = self.fc_shape(input_size)
        self.fc_shape2 = self.fc_shape(seq_size)
        self.conv1 = nn.Conv2d(1, 6, self.kernel_size)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.p = p
        self.fc   = nn.Sequential(
            nn.Linear(16*self.fc_shape1*self.fc_shape2, 1024),
            nn.ReLU(),
            nn.Dropout(p=self.p),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=self.p),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=self.p),
            nn.Linear(512, self.output_size)
        )
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def fc_shape(self, in_size):
        fc_shape = int((in_size-self.kernel_size+1-2)/2 + 1)
        fc_shape = int((fc_shape-self.kernel_size+1-2)/2 + 1)
        return fc_shape

class MultimodalLeNet(nn.Module):
    '''
    LeNet model for Multimodal datasets.
    '''
    def __init__(self, in1, in2, in3, seq, LeNet_output=512, p=0.05):
        super(MultimodalLeNet, self).__init__()
        self.in1 = in1/seq
        self.in2 = in2/seq
        self.in3 = in3/seq
        self.seq = seq
        self.p = p
        self.LeNet_output = LeNet_output
        self.LeNet1 = LeNet(self.in1, self.seq, self.LeNet_output, p=p)
        self.LeNet2 = LeNet(self.in2, self.seq, self.LeNet_output, p=p)
        self.LeNet3 = LeNet(self.in3, self.seq, self.LeNet_output, p=p)
        self.fc = nn.Sequential(
                nn.Linear(self.LeNet_output*3, LeNet_output),
                nn.ReLU(),
                nn.Dropout(p=self.p),
                nn.BatchNorm1d(LeNet_output),
                nn.Linear(LeNet_output, 256),
                nn.ReLU(),
                nn.Dropout(p=self.p),
                nn.BatchNorm1d(256),
                nn.Linear(256, 84),
                nn.ReLU(),
                nn.Dropout(p=self.p),
                nn.BatchNorm1d(84),
                nn.Linear(84, 6),
                nn.Sigmoid(),
        )
    def forward(self, x1, x2, x3):
        out1 = self.LeNet1(x1)
        out2 = self.LeNet2(x2)
        out3 = self.LeNet3(x3)
        out = torch.cat([out1, out2, out3], dim=1)
        out = self.fc(out)
        return out
