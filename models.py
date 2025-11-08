import torch.nn as nn

class SimpleGRU(nn.Module):
    """
    A simple GRU-based model for sequence regression tasks.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SimpleGRU, self).__init__()
        self.gru = nn.GRU(input_size=input_size, 
                          hidden_size=hidden_size, 
                          num_layers=num_layers, 
                          batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch_size, sequence_length, input_size)
        out, _ = self.gru(x)  # out: (batch_size, sequence_length, hidden_size)
        out = out[:, -1, :]   # Take the last time step's output
        out = self.relu(self.fc1(out))    # out: (batch_size, output_size)
        return self.fc2(out)