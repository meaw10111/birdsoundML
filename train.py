import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models
from tqdm import tqdm
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# Define transformations
transform_train = transforms.Compose([
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data_path = r"C:\Users\ACER\Desktop\meaw\dtafafc\train" # Path to training data folder
test_data_path = r"C:\Users\ACER\Desktop\meaw\dtafafc\test" 
# Create datasets and dataloaders
train_dataset = ImageFolder(root=train_data_path , transform=transform_train)
test_dataset = ImageFolder(root=test_data_path, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4)

# Define DenseNet-201 model
model = models.densenet201(pretrained=True)
model.to(device)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, len(train_dataset.classes))  # Number of classes is inferred from train_dataset

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
model.train()
for epoch in range(10):  # Adjust number of epochs as needed
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"): # tqdm added here
        inputs, labels = inputs.to(device), labels.to(device) 
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss}")
    

# Save the trained model
torch.save(model.state_dict(), r"C:\Users\ACER\Desktop\meaw\firstmodel.pth")
