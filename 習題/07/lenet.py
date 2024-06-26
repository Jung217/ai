# 參考 https://github.com/ccc112b/py2cs/blob/master/03-人工智慧/06-強化學習/01-強化學習/01-gym/04-run/cartpole_human_run.py
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):    
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 150, kernel_size=5)
        self.conv2 = nn.Conv2d(150, 10, kernel_size=5)
        #self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(160, 80)
        self.fc2 = nn.Linear(80, 40)
        self.fc3 = nn.Linear(40, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 160)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x