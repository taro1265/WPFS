from torch import nn
import torch.nn.functional as F
from torchnmf.nmf import NMF
import torch
import torch.optim as optim

class EmbeddingNetwork(nn.Module):
    def __init__(self, args, input_matrix):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_matrix = input_matrix.to(device)
        super().__init__()

        nmf = NMF(input_matrix.shape, rank=args.wpn_embedding_size).cuda()
        nmf.fit(input_matrix.cuda(), beta=2, max_iter=1000, verbose=True)
        input_pinv = torch.linalg.pinv(input_matrix)
        input_init = torch.matmul(input_pinv, nmf.H)

        self.args = args
        self.register_buffer('input_matrix', input_matrix) # store the static embedding_matrix

        self.M = args.wpn_embedding_size
        self.N = input_matrix.shape[1]
        self.K = input_matrix.shape[0]
        self.H = 100
        self.dropout_prob = 0.5
        model = NMFApproximator(input_size=self.N, output_size=self.M, init_weight=input_init)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        criterion = nn.MSELoss()
        train_model(model, optimizer, scheduler, criterion, input_matrix, nmf.H, epochs=1000)

        self.fc1 = model
        self.bn1 = nn.BatchNorm1d(self.M)
        self.dropout1 = nn.Dropout(self.dropout_prob)
        self.fc2 = nn.Linear(self.M, self.H)
        self.bn2 = nn.BatchNorm1d(self.H)
        self.dropout2 = nn.Dropout(self.dropout_prob)
        self.fc3 = nn.Linear(self.H, self.M)

        #self.fc1.weight.data = input_init.T.float()
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu')

    def forward(self):
        x = self.fc1(self.input_matrix)
        return x

class NMFApproximator(nn.Module):
    def __init__(self, input_size, output_size, init_weight):
        super(NMFApproximator, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.fc1.weight.data = init_weight.T.float()
        self.fc2 = nn.Linear(output_size, output_size)
        nn.init.eye_(self.fc2.weight)


    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x

def train_model(model, optimizer, scheduler, criterion, input_matrix, target_matrix, epochs):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        output_matrix = model(input_matrix)

        # Compute loss
        loss = criterion(output_matrix, target_matrix)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')