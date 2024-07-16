import torch

class TwoLayerNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate, seed):
        super(TwoLayerNN, self).__init__()
        torch.manual_seed(seed)
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2_1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 4)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.dropout = torch.nn.Dropout(dropout_rate)
        
    def forward(self, x, beta=0):
        
        y = self.dropout(self.relu((self.fc1(x))))
        # y = y +  beta*abs(y).mean() * torch.randn(y.shape)
        y = self.dropout(self.relu((self.fc2(y))))
        
        y = self.fc3(y)
        return y
    
    
