import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import sys

BATCH_SIZE = 32
EPOCH = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.001
momentum = 0.9

transform = transforms.Compose([
    transforms.Resize(227),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = torchvision.datasets.CIFAR10(root='./data/', train=True, transform=transform)
test_data = torchvision.datasets.CIFAR10(root='./data/', train=False, transform=transform)

train_loader = Data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = Data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

classes_names = train_data.classes


class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, padding=2, groups=2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, padding=1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, padding=1, groups=2),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, padding=1, groups=2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.nn1 = nn.Sequential(
            nn.Linear(256*6*6, 4096),
            nn.ReLU(),
            nn.Dropout()
        )
        self.nn2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout()
        )
        self.out = nn.Linear(4096, num_classes)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0),256*6*6)
        x = self.nn1(x)
        x = self.nn2(x)
        x = self.out(x)
        return x


def train(model, device, train_loader, optimizer, loss_func, epoch):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        output = model(inputs)
        loss = loss_func(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.data.cpu().numpy()
        if i % 200 == 199:
            print('[%d,%5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0
            
    
def test(model, device, train_loader, test_loader):
    accuracy = 0
    total = 0
    with torch.no_grad():
        for data in train_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    
    print('Accuracy of the network on the 50000 train images: %d %%' % (100 * accuracy / total))

    accuracy = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    
    print('Accuracy of the network on the 10000 test images: %d %%' % (100*accuracy/total))

    class_correct = [0.0]*10
    class_total = [0.0]*10
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs.data, 1)
            c = (predicted == labels)
            if len(c) == 16:
                for i in range(16):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += c[i].item()
            else:
                for i in range(32):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
    
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes_names[i], 100*class_correct[i]/class_total[i]))


if __name__ == '__main__':
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print('usage: python AlexNet.py <train/test> <model_path>')
        sys.exit(1)

    if sys.argv[1] == 'train':
        dataiter = iter(train_loader)
        images, labels = next(dataiter)
        print(' '.join('%5s' % classes_names[labels[j]] for j in range(BATCH_SIZE)))

        model = AlexNet().to(DEVICE)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        loss_func = nn.CrossEntropyLoss()
        for epoch in range(0, EPOCH):
            train(model, DEVICE, train_loader, optimizer, loss_func, epoch)
            
        print('Finished Training')
        torch.save(model, sys.argv[2])

    elif sys.argv[1] == 'test':
        model = torch.load(sys.argv[2])
        
        test(model, DEVICE, train_loader, test_loader)
