
import torch
from model import MLP

model = MLP()

# Simulate a batch of 5 records, each with 12 features
sample_input = torch.randn(5, 12)

output = model(sample_input)

print("Model:")
print(model)

print("\nInput shape:", sample_input.shape)
print("Output shape:", output.shape)
print("Output:", output)
