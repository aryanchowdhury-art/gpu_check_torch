import torch
import torch.nn as nn
import torch.optim as optim

# Linear Regression Model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # Input: 1 feature, Output: 1 target

    def forward(self, x):
        return self.linear(x)

# Generate Data
x = torch.linspace(0, 10, 100).view(-1, 1)  # Inputs: 100 points from 0 to 10
y = 3 * x + 7 + torch.randn(x.size())       # True relation: y = 3x + 7 (with noise)

# Initialize Model, Loss, Optimizer
model = LinearRegression()
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training Loop
for epoch in range(1000):
    # Forward Pass
    predictions = model(x)
    loss = criterion(predictions, y)

    # Backward Pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/1000], Loss: {loss.item():.4f}")

# Final Parameters
print("Learned Parameters:", list(model.parameters()))
