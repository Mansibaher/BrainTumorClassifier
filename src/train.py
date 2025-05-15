import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

from utils import get_dataloaders

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

# Load data
train_loader, val_loader = get_dataloaders("data", batch_size=32)

# Load pretrained ResNet18
model = models.resnet18(weights="IMAGENET1K_V1")
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # Binary classification
model = model.to(device)

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"🔁 Epoch {epoch+1}: Loss = {avg_loss:.4f}")

# Save model
torch.save(model.state_dict(), "models/model.pth")
print("✅ Model saved to models/model.pth")
