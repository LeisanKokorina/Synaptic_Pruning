import logging

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.metrics import precision_recall_fscore_support
import os
import time
from torch.optim.lr_scheduler import StepLR

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] {%(filename)s:%(lineno)d}'
           ' %(levelname)s\n%(message)s',
    datefmt='%H:%M:%S'
)

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    logger.info(f"Shape of X [N, C, H, W]: {X.shape}")
    logger.info(f"Shape of y: {y.shape} {y.dtype}")
    break

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
logger.info(f"Using {device} device")


class ADLNet(nn.Module):
    def __init__(self, num_classes=100):
        super(ADLNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 128 * 3 * 3)
        x = self.classifier(x)
        return x


model = ADLNet().to(device)
# print(model.summary())
logger.info(model)
lr_step_gamma = 0.7
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=1, gamma=lr_step_gamma)


def train(dataloader, model, loss_fn, optimizer, scheduler, prune_percentage):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Get the magnitudes of weights and sort them in descending order
        weight_magnitudes = []
        for param in model.parameters():
            weight_magnitudes.extend(torch.abs(param.data).flatten())
        weight_magnitudes = torch.sort(torch.tensor(weight_magnitudes), descending=True).values

        # Set pruning threshold based on specified percentage
        prune_threshold = weight_magnitudes[int(prune_percentage * len(weight_magnitudes))]

        # Prune weights based on the threshold
        for param in model.parameters():
            mask = torch.abs(param.data) >= prune_threshold
            param.data *= mask

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            logger.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        # Update the learning rate
        scheduler.step()


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    logger.info(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# Fine-tuning after pruning
fine_tune_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Use lower learning rate
fine_tune_epochs = 2
prune_percentage = 0.2
for t in range(fine_tune_epochs):
    logger.info(f"Fine-tuning Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, fine_tune_optimizer, scheduler, prune_percentage)
    test(test_dataloader, model, loss_fn)

# Main training loop
epochs = 5
start_time = time.time()
for t in range(epochs):
    logger.info(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer, scheduler, prune_percentage)
    test(test_dataloader, model, loss_fn)
end_time = time.time()
training_time = end_time - start_time
logger.info('Finished Training')
logger.info('Training time: ', training_time, 'seconds')
logger.info("Done!")

model.to(device)


def calculate_metrics(model, dataloader):
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

    accuracy = correct / total
    precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    return accuracy, precision, recall


# Calculate metrics
accuracy, precision, recall = calculate_metrics(model, test_dataloader)

logger.info(
    f'Accuracy: {accuracy}, '
    f'Precision: {precision}, '
    f'Recall: {recall}'
)


def get_model_size(model):
    torch.save(model.state_dict(), "model.pth")
    size_MB = os.path.getsize("model.pth") / 1e6
    return size_MB


def get_inference_time(model, input_shape=(1, 1, 28, 28), repeat=100):
    device = next(model.parameters()).device  # Get the device of the model
    model.eval()
    input_data = torch.randn(input_shape).to(device)  # Move input data to the same device as the model
    start_time = time.time()
    for _ in range(repeat):
        with torch.no_grad():
            _ = model(input_data)
    return (time.time() - start_time) / repeat


# Get model size
logger.info(
    f'Model size: {get_model_size(model)} MB; '
    f'Average inference time: {get_inference_time(model)} s'  # Get inference time
)

model = ADLNet()
model.load_state_dict(torch.load("model.pth"))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
x = x.unsqueeze(0)
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    logger.info(f'Predicted: "{predicted}", Actual: "{actual}"')
