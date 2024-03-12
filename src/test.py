import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureSelector(nn.Module):
	def __init__(self, temperature=0.5):
		super().__init__()
		self.temperature = temperature
	def forward(self, x):
		batch_size = x.size(0)
		input_features = x.size(1)
		gumbel_noise0 = -torch.log(-torch.log(torch.rand(batch_size, input_features)))
		gumbel_noise1 = -torch.log(-torch.log(torch.rand(batch_size, input_features)))
		logits = torch.stack([torch.log(x)+gumbel_noise1+torch.rand(batch_size, input_features), torch.log(1-x)+gumbel_noise0], dim=0)
		print(logits)
		mask = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=0)
		return x * mask[0]

input_features = 10
model = FeatureSelector()

x = torch.rand(2, input_features)
print(x)
selected_features = model(x)
print(selected_features)
print(torch.max(selected_features)-x)