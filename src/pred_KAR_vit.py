import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTFeatureExtractor
import os
from skimage.util import view_as_windows


# Set device
device = torch.device("cpu")

# Initialize the model
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                  num_labels=3)

# Load the saved model
model.load_state_dict(torch.load('/opt/project/tmp/vit_model1.pth'))
model = model.to(device)
model.eval()

# Load the test dataset
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = '/opt/project/dataset/KAR_training' 

test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'KAR_testing'), test_transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

def patch_image(image):
    width, height = image.size
    patch_width = width // 3
    patch_height = height // 3

    patches = []
    for i in range(3):
        for j in range(3):
            left = i * patch_width
            upper = j * patch_height
            right = left + patch_width
            lower = upper + patch_height

            patch = image.crop((left, upper, right, lower))
            patches.append(patch)

    return patches

def display_patches(patches):
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))

    for i, patch in enumerate(patches):
        ax = axs[i // 3, i % 3]
        ax.imshow(patch)
        ax.axis('off')

    plt.show()

def pred_patches(model, patches, all_preds_patches):
    for patch in patches:
            patch = test_transform(patch)
            patch = patch.unsqueeze(0).to(device)
            outputs = model(patch)['logits']
            _, pred = torch.max(outputs, 1)
            all_preds_patches.append(pred.item())

# Function to make predictions
# def predict(model, dataloader):
#     model.eval()
#     all_preds = []
#     all_labels = []
#     with torch.no_grad():
#         for inputs, labels in dataloader:
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             outputs = model(inputs)['logits']
#             _, preds = torch.max(outputs, 1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
#     return all_labels, all_preds

# # Make predictions
# true_labels, pred_labels = predict(model, test_loader)

# # Calculate evaluation metrics
# accuracy = accuracy_score(true_labels, pred_labels)
# precision = precision_score(true_labels, pred_labels, average='macro')
# recall = recall_score(true_labels, pred_labels, average='macro')
# f1 = f1_score(true_labels, pred_labels, average='macro')

# Function to make predictions
def predict(model, image_paths, transform):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for image_path in image_paths:
            # Load image
            image = Image.open(image_path)
            patches = patch_image(Image.open(image_path))
            # display_patches(patches)

            
            # Get label from image path
            label = image_path.split('/')[-2]
            if label == 'Heavy':
                label = 0
            elif label == 'Moderate':
                label = 1
            else:
                label = 2
                
            # Preprocess image
            image = test_transform(image)
            image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device
            
            # Predict
            outputs = model(image)['logits']
            _, pred = torch.max(outputs, 1)
            
            if pred.item() == 0:
            
                all_preds_patches = []
                
                for patch in patches:
                    patch = transform(patch)
                    patch = patch.unsqueeze(0).to(device)
                    outputs = model(patch)['logits']
                    _, pred = torch.max(outputs, 1)
                    all_preds_patches.append(pred.item())
                
                sum_preds = sum(all_preds_patches)/9
                
                if sum_preds < 0.75:
                    all_preds.append(0)
                elif sum_preds < 1.85:
                    all_preds.append(1)
                else:
                    all_preds.append(2)
            else:
                all_preds.append(pred.item())
            
            
            all_labels.append(label)
            
            # print(f'Prediction for this image: {pred.item()}')
    
    return all_labels, all_preds

# Get list of image paths
image_paths = glob.glob(os.path.join(data_dir, 'KAR_testing', '*', '*.jpg'))  # Modify the pattern as needed

# Make predictions
true_labels, pred_labels = predict(model, image_paths, test_transform)

# Calculate evaluation metrics
accuracy = accuracy_score(true_labels, pred_labels)
precision = precision_score(true_labels, pred_labels, average='macro')
recall = recall_score(true_labels, pred_labels, average='macro')
f1 = f1_score(true_labels, pred_labels, average='macro')

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Plot confusion matrix
conf_matrix = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
plt.title('Confusion Matrix')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.savefig('/opt/project/tmp/confusion_matrix_post8.png')
