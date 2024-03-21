import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models
import os
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Define transformations
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_data_path = r"C:\Users\ACER\Desktop\meaw\dtafafc\test" 

# Create dataset and dataloader for testing
test_dataset = ImageFolder(root=test_data_path, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=4)

# Define DenseNet-201 model
model = models.densenet201(pretrained=False)  # Ensure pretrained is False for testing
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, len(test_dataset.classes))
model.to(device)

# Load the trained model
model.load_state_dict(torch.load(r"C:\Users\ACER\Desktop\meaw\firstmodel.pth"))
model.eval()  # Set model to evaluation mode

# Test the model
correct = 0
total = 0
predicted_labels = []
true_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predicted_labels.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

accuracy = 100 * correct / total
print('Accuracy of the network on the test images: {:.2f}%'.format(accuracy))

# Print prediction result for each image
print("\nPrediction Results:")
for i, (predicted, true) in enumerate(zip(predicted_labels, true_labels)):
    print("Image {}: Predicted - {}, Actual - {}".format(i+1, test_dataset.classes[predicted], test_dataset.classes[true]))

# Generate confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)
print("\nConfusion Matrix:")
print(conf_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')

plt.savefig (r"C:\Users\ACER\Desktop\meaw\confusion_matrix.png", bbox_inches='tight')
plt.show()
