
# 1. load and normalize the CIFAR10 training and test datasets using torchvision
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(
    root = './data', 
    train = True, 
    download = True, 
    transform = transform
    )

trainloader = torch.utils.data.DataLoader(
    trainset, 
    batch_size = batch_size, 
    shuffle = True, 
    num_workers = 0
    )

testset = torchvision.datasets.CIFAR10(
    root = './data', 
    train = False, 
    download = True, 
    transform = transform
    )

testloader = torch.utils.data.DataLoader(
    testset, 
    batch_size = batch_size, 
    shuffle = False, 
    num_workers = 0
    )

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
    # unnormalize the image
    img = img/2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

#show images
#imshow(torchvision.utils.make_grid(images))

#print labels
#print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


# 2. Deine a CNN

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten all the dimensions except batch
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

# 3. Define a loss funciton

#Let's use a Classification Cross-Entropy loss and SGD with momentum = 0.9
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

# 4. Train the network on the training data

for epoch in range(2):
    
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # Get the inputs, as data is a list of [inputs, labels]
        inputs, labels = data

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print("Finished Training")


PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)


# 5. Test the network on the test data

"""
# Display an image from the test set
dataiter2 = iter(testloader)
images, labels = next(dataiter2)

#print images
#imshow(torchvision.utils.make_grid(images))
#print('Ground Truth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))


# Test the performance on a small subset of the dataset

_, predicted = torch.max(outputs, 1)

print('Predicted: ', 

' '.join(f'{classes[predicted[j]]:5s}' 
    for j in range(4))
    )
"""

# Let's see how the dataset performs on the entire dataset

correct = 0
total = 0

# We don't need to calculate the gradients since we are not training
with torch.no_grad():
    for data in testloader:
        images, labels = data
        
        #calculate the outputs by running the images through the network
        outputs = net(images)

        #The class with the highest energy is chosen as the prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct //total} %')

# Prepare to count the predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# Again we don't need gradients
with torch.no_grad():
    for data in testloader:
        images, labels = data 
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # Collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

for classname, correct_count in correct_pred.items():
    accuracy = 100*float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)