import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
from functools import reduce
import sys

BATCH_SIZE = 512
EPOCH = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 0.001


class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.nn1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.nn2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.out = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, reduce(lambda x, y: x * y, x.size()[1:]))
        x = self.nn1(x)
        x = self.nn2(x)
        x = self.out(x)
        return x


transforms = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transforms)
test_data = torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=transforms)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


def train(model, device, train_loader, optimizer, loss_func, epoch):

    model.train()
    for step, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = loss_func(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 30 == 0:
            print('Epoch: {} [{}/{} ({:.0f}%]\tLoss: {:.6f}'.format(
                epoch, step * len(data), len(train_loader.dataset),
                       100. * step / len(train_loader), loss.data.cpu().numpy()))


def test(model, device, test_loader, loss_func):

    model.eval()
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_func(output, target)
            pred = torch.max(output, 1)[1].to(device).data
            accuracy += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, accuracy, len(test_loader.dataset),
        100. * accuracy / len(test_loader.dataset)))


if __name__ == '__main__':

    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print('usage: python LeNet5.py <train/test> <model_path>')
        sys.exit(1)

    if sys.argv[1] == 'train':
        model = LeNet5().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        loss_func = nn.CrossEntropyLoss(reduction='sum')
        for epoch in range(EPOCH):
            train(model, DEVICE, train_loader, optimizer, loss_func, epoch)
        torch.save(model, sys.argv[2])
        print('Training Done.')

    elif sys.argv[1] == 'test':
        model = torch.load(sys.argv[2])
        loss_func = nn.CrossEntropyLoss(reduction='sum')
        test(model, DEVICE, test_loader, loss_func)
