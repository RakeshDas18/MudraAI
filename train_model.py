import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import os
from skimage.io import imread
from skimage.transform import resize
from PIL import Image
from sklearn.model_selection import train_test_split


Categories = ['alopodmo', 'ankush', 'ardhachandra', 'bhramar', 'chatur', 'ghronik', 'hongshashyo', 
              'kangul', 'kodombo', 'kopitho', 'krishnaxarmukh', 'mrigoshirsho', 'mukul', 'unknown']
datadir = 'IMAGES/train'
unknown_datadir = 'IMAGES/unknown' 

# Define Image Dataset Class
class MudraDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        image = Image.fromarray((image * 255).astype(np.uint8))  # Convert back to uint8
        
        if self.transform:
            image = self.transform(image)

        return image, label

# Data Augmentation
transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

# Load and Process Images
def load_images(data_path, class_label):
    images, labels = [], []
    for img_name in os.listdir(data_path):
        img_array = imread(os.path.join(data_path, img_name))
        img_resized = resize(img_array, (150, 150, 3))
        images.append(img_resized)
        labels.append(class_label)
    return images, labels

# Load Known Mudra Images
data, labels = [], []
for i, category in enumerate(Categories[:-1]):  # Exclude 'Unknown' for now
    path = os.path.join(datadir, category)
    img_data, img_labels = load_images(path, i)
    data.extend(img_data)
    labels.extend(img_labels)
    print(f'Loaded category: {category} successfully')


# Load Unknown Images
unknown_images, unknown_labels = load_images(unknown_datadir, len(Categories) - 1)  # Assign last index to "Unknown"
data.extend(unknown_images)
labels.extend(unknown_labels)

# Convert to Numpy Arrays
data = np.array(data, dtype=np.float32)
labels = np.array(labels)

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels)

# Create Datasets & Loaders
train_dataset = MudraDataset(x_train, y_train, transform=transform)
test_dataset = MudraDataset(x_test, y_test, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the Updated SVM Model
class SVM(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SVM, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)  # Flatten
        return self.fc(x)

# Load the Saved Model and Modify for 14 Classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 150 * 150 * 3
num_classes = len(Categories)  # Now 14 (13 known + 1 unknown)

# model = SVM(input_dim, num_classes).to(device)
saved_model_path = 'mudra_model_7.pth'
# Load original 13-class model first
original_model = SVM(input_dim, 13).to(device)
original_model.load_state_dict(torch.load(saved_model_path, map_location=device))

# Create new model with 14 classes
model = SVM(input_dim, 14).to(device)

# Transfer weights (excluding the final layer)
with torch.no_grad():
    model.fc.weight[:13] = original_model.fc.weight
    model.fc.bias[:13] = original_model.fc.bias

# Define Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

# Retrain the Model with "Unknown" Class
epochs = 20
for epoch in range(epochs):
    model.train()
    running_loss, correct_train, total_train = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct_train / total_train

    # Validation Phase
    model.eval()
    val_running_loss, correct_val, total_val = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

    val_loss = val_running_loss / len(test_loader)
    scheduler.step(val_loss)
    val_accuracy = correct_val / total_val

    print(f'Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Train Acc={train_accuracy:.4f}, Val Acc={val_accuracy:.4f}')

# Save the Updated Model
torch.save(model.state_dict(), 'mudra_model_8_2.pth')

# Evaluate Model with Unknown Class
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Compute Accuracy
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy = accuracy_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)

print(f"\nModel Accuracy: {accuracy*100:.2f}%")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=Categories))

# Plot Confusion Matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=Categories, yticklabels=Categories)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix with Unknown Class')
plt.show()

# Identify misclassified samples
misclassified_indices = np.where(np.array(y_true) != np.array(y_pred))[0]

# Print misclassified instances
print("\nMisclassified Samples:")
for i in misclassified_indices:
    print(f"True: {Categories[y_true[i]]}, Predicted: {Categories[y_pred[i]]}")