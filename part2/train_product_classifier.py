import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
BATCH_SIZE = 64
HEAD_EPOCHS = 10
FINETUNE_EPOCHS = 3
HEAD_LR = 1e-3
FINETUNE_LR = 1e-4
VAL_SIZE = 5000
NUM_CLASSES = 10

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "fashion_mnist")
SAMPLE_DIR = os.path.join(ROOT, "data", "sample_images")
MODEL_DIR = os.path.join(ROOT, "models")
CACHE_DIR = os.path.join(ROOT, "data", "feature_cache")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("FLIPKART PART 2 - PRODUCT IMAGE CATEGORISER")
print("=" * 70)
print("Device:", device)

# ============================================================
# FASHION-MNIST LABELS
# ============================================================

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# ============================================================
# TRANSFORMS
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# A simple transform for exporting the original images.
raw_transform = transforms.ToTensor()

# ============================================================
# LOAD FASHION-MNIST
# ============================================================

print("\nDownloading/loading Fashion-MNIST...")

full_train = datasets.FashionMNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform
)

# Separate dataset without transformed images for PNG export.
raw_test_dataset = datasets.FashionMNIST(
    root=DATA_DIR,
    train=False,
    download=False
)

labels = np.array(full_train.targets)

all_indices = np.arange(len(full_train))

train_indices, val_indices = train_test_split(
    all_indices,
    test_size=VAL_SIZE,
    random_state=SEED,
    stratify=labels
)

train_dataset = Subset(full_train, train_indices)
val_dataset = Subset(full_train, val_indices)

print("\nDATASET SPLIT")
print("-" * 50)
print("Training images:   ", len(train_dataset))
print("Validation images: ", len(val_dataset))
print("Test images:       ", len(test_dataset))

# ============================================================
# DATALOADERS
# ============================================================

num_workers = 0

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers
)

# ============================================================
# PRETRAINED RESNET-18
# ============================================================

print("\nLoading pretrained ResNet-18...")

weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Freeze entire pretrained backbone.
for parameter in model.parameters():
    parameter.requires_grad = False

# Replace ImageNet classifier with Fashion-MNIST classifier.
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, NUM_CLASSES)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=HEAD_LR
)

# ============================================================
# FEATURE EXTRACTION / CACHING
# ============================================================

print("\nExtracting and caching frozen ResNet features...")
print("This may take some time on a CPU.")

feature_cache_file = os.path.join(
    CACHE_DIR,
    "resnet18_fashion_features.pt"
)

def extract_features(loader, split_name):
    feature_list = []
    label_list = []

    model.eval()

    with torch.no_grad():

        for batch_num, (images, labels_batch) in enumerate(loader):

            images = images.to(device)

            # Remove classifier temporarily by using ResNet backbone.
            features = model.avgpool(
                model.layer4(
                    model.layer3(
                        model.layer2(
                            model.layer1(
                                model.maxpool(
                                    model.relu(
                                        model.bn1(
                                            model.conv1(images)
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )

            features = torch.flatten(features, 1)

            feature_list.append(features.cpu())
            label_list.append(labels_batch)

            if (batch_num + 1) % 50 == 0:
                print(
                    f"{split_name}: "
                    f"{batch_num + 1}/{len(loader)} batches processed"
                )

    return torch.cat(feature_list), torch.cat(label_list)


if os.path.exists(feature_cache_file):

    print("Loading cached features...")

    cache = torch.load(
        feature_cache_file,
        map_location="cpu"
    )

    train_features = cache["train_features"]
    train_labels = cache["train_labels"]

    val_features = cache["val_features"]
    val_labels = cache["val_labels"]

    test_features = cache["test_features"]
    test_labels = cache["test_labels"]

else:

    train_features, train_labels = extract_features(
        train_loader,
        "TRAIN"
    )

    val_features, val_labels = extract_features(
        val_loader,
        "VALIDATION"
    )

    test_features, test_labels = extract_features(
        test_loader,
        "TEST"
    )

    torch.save(
        {
            "train_features": train_features,
            "train_labels": train_labels,
            "val_features": val_features,
            "val_labels": val_labels,
            "test_features": test_features,
            "test_labels": test_labels
        },
        feature_cache_file
    )

    print("\nFeature cache saved.")

# ============================================================
# TRAIN CLASSIFIER HEAD
# ============================================================

print("\n" + "=" * 70)
print("TRAINING CLASSIFIER HEAD")
print("=" * 70)

head_train_loader = DataLoader(
    torch.utils.data.TensorDataset(
        train_features,
        train_labels
    ),
    batch_size=256,
    shuffle=True
)

head_val_loader = DataLoader(
    torch.utils.data.TensorDataset(
        val_features,
        val_labels
    ),
    batch_size=256,
    shuffle=False
)

best_val_accuracy = 0.0
best_head_state = None

for epoch in range(HEAD_EPOCHS):

    model.train()

    # Only classifier head is trainable.
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    running_loss = 0.0

    for features_batch, labels_batch in head_train_loader:

        features_batch = features_batch.to(device)
        labels_batch = labels_batch.to(device)

        optimizer.zero_grad()

        outputs = model.fc(features_batch)

        loss = criterion(
            outputs,
            labels_batch
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for features_batch, labels_batch in head_val_loader:

            features_batch = features_batch.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model.fc(features_batch)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels_batch
            ).sum().item()

            total += labels_batch.size(0)

    val_accuracy = correct / total

    print(
        f"Epoch {epoch + 1}/{HEAD_EPOCHS} "
        f"| Loss: {running_loss / len(head_train_loader):.4f} "
        f"| Validation Accuracy: {val_accuracy:.4f}"
    )

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_head_state = {
            key: value.cpu().clone()
            for key, value in model.state_dict().items()
        }

# Restore best classifier-head model.
if best_head_state is not None:
    model.load_state_dict(best_head_state)

print(
    f"\nFeature-extraction validation accuracy: "
    f"{best_val_accuracy:.4f}"
)

before_finetune_accuracy = best_val_accuracy

# ============================================================
# OPTIONAL FINE-TUNING
# ============================================================

after_finetune_accuracy = before_finetune_accuracy
fine_tuning_used = False

if before_finetune_accuracy < 0.80:

    print("\nValidation accuracy is below 80%.")
    print("Fine-tuning late ResNet layers...")

    fine_tuning_used = True

    # Unfreeze only late layer4 and classifier.
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    optimizer_ft = torch.optim.Adam(
        [
            {
                "params": model.layer4.parameters(),
                "lr": FINETUNE_LR
            },
            {
                "params": model.fc.parameters(),
                "lr": FINETUNE_LR
            }
        ]
    )

    # Fine-tuning needs real images, not cached features.
    train_loader_ft = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader_ft = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers
    )

    best_ft_accuracy = before_finetune_accuracy
    best_ft_state = None

    for epoch in range(FINETUNE_EPOCHS):

        model.train()

        running_loss = 0.0

        for images, labels_batch in train_loader_ft:

            images = images.to(device)
            labels_batch = labels_batch.to(device)

            optimizer_ft.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels_batch
            )

            loss.backward()
            optimizer_ft.step()

            running_loss += loss.item()

        # Validation
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels_batch in val_loader_ft:

                images = images.to(device)
                labels_batch = labels_batch.to(device)

                outputs = model(images)

                predictions = torch.argmax(
                    outputs,
                    dim=1
                )

                correct += (
                    predictions == labels_batch
                ).sum().item()

                total += labels_batch.size(0)

        val_accuracy = correct / total

        print(
            f"Fine-tune epoch {epoch + 1}/{FINETUNE_EPOCHS} "
            f"| Loss: {running_loss / len(train_loader_ft):.4f} "
            f"| Validation Accuracy: {val_accuracy:.4f}"
        )

        if val_accuracy > best_ft_accuracy:

            best_ft_accuracy = val_accuracy

            best_ft_state = {
                key: value.cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_ft_state is not None:
        model.load_state_dict(best_ft_state)

    after_finetune_accuracy = best_ft_accuracy

else:

    print("\nFeature extraction alone achieved at least 80%.")
    print("Fine-tuning is NOT required.")

print(
    "\nValidation accuracy before fine-tuning: "
    f"{before_finetune_accuracy:.4f}"
)

print(
    "Validation accuracy after fine-tuning:  "
    f"{after_finetune_accuracy:.4f}"
)

# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

model.eval()

all_predictions = []
all_true = []

# If fine-tuning happened, use real images.
if fine_tuning_used:

    evaluation_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers
    )

    with torch.no_grad():

        for images, labels_batch in evaluation_loader:

            images = images.to(device)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_true.extend(
                labels_batch.numpy()
            )

else:

    with torch.no_grad():

        outputs = model.fc(
            test_features.to(device)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions = predictions.cpu().numpy()
        all_true = test_labels.numpy()

all_predictions = np.array(all_predictions)
all_true = np.array(all_true)

test_accuracy = accuracy_score(
    all_true,
    all_predictions
)

cm = confusion_matrix(
    all_true,
    all_predictions
)

print(
    f"\nFINAL TEST ACCURACY: "
    f"{test_accuracy:.4f}"
)

print("\nCONFUSION MATRIX")
print(cm)

print("\nPER-CLASS PRECISION / RECALL")
print(
    classification_report(
        all_true,
        all_predictions,
        target_names=CLASS_NAMES,
        digits=4
    )
)

# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "product_classifier.pt"
)

torch.save(
    model.state_dict(),
    model_path
)

print(
    "\nModel saved to:",
    model_path
)

# ============================================================
# EXPORT REAL TEST IMAGES
# ============================================================

print("\nExporting real Fashion-MNIST test images...")

# Pick examples from different classes.
selected = {}

for index in range(len(raw_test_dataset)):

    image, label = raw_test_dataset[index]

    label = int(label)

    if label not in selected:

        selected[label] = (
            index,
            image
        )

    if len(selected) == 5:
        break

for label, (index, image) in selected.items():

    filename = (
        f"{label:02d}_"
        f"{CLASS_NAMES[label].lower().replace('/', '_').replace(' ', '_')}"
        f".png"
    )

    output_path = os.path.join(
        SAMPLE_DIR,
        filename
    )

    image.save(output_path)

    print(
        f"Saved: {output_path}"
    )

# ============================================================
# SAVE METADATA
# ============================================================

metadata = {
    "model": "ResNet-18",
    "dataset": "Fashion-MNIST",
    "dataset_source": "Zalando Research Fashion-MNIST",
    "input_size": "224x224",
    "channels": 3,
    "imagenet_mean": [0.485, 0.456, 0.406],
    "imagenet_std": [0.229, 0.224, 0.225],
    "batch_size": BATCH_SIZE,
    "optimizer": "Adam",
    "head_learning_rate": HEAD_LR,
    "fine_tune_learning_rate": FINETUNE_LR,
    "head_epochs": HEAD_EPOCHS,
    "fine_tune_epochs": FINETUNE_EPOCHS,
    "train_size": len(train_dataset),
    "validation_size": len(val_dataset),
    "test_size": len(test_dataset),
    "feature_extraction_validation_accuracy": float(
        before_finetune_accuracy
    ),
    "final_validation_accuracy": float(
        after_finetune_accuracy
    ),
    "test_accuracy": float(
        test_accuracy
    ),
    "fine_tuning_used": fine_tuning_used,
    "class_names": CLASS_NAMES
}

metadata_path = os.path.join(
    MODEL_DIR,
    "product_classifier_metadata.json"
)

with open(
    metadata_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=2
    )

# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("PART 2 TRAINING COMPLETE")
print("=" * 70)

print(
    "Test accuracy:",
    round(test_accuracy, 4)
)

print(
    "Model:",
    model_path
)

print(
    "Sample images:",
    SAMPLE_DIR
)