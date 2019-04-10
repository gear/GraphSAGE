import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# graph_sage modules
from utils import load_data as gs_load_data

# other helpers
from sklearn import metrics
import numpy as np
import argparse


def calc_f1(y_true, y_pred):
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    return metrics.f1_score(y_true, y_pred, average="micro"),\
           metrics.f1_score(y_true, y_pred, average="macro")


class FeaturesOnlyDataset(Dataset):
    def __init__(self, prefix, transforms=None):
        """
        Args:
            prefix (string): path to data prefixes
        """
        data = gs_load_data(prefix)
        G, feats, id_map, _, class_map = data
        self.num_samples = len(id_map)
        self.train_idx = []
        self.val_idx = []
        self.test_idx = []
        class_sample_val = class_map.itervalues().next()
        if isinstance(class_sample_val, list):
            self.labels = np.zeros((self.num_samples, len(class_sample_val)))
        else:
            self.labels = np.zeros((self.num_samples, 1))
        for nodeid in G.nodes():
            self.labels[id_map[nodeid]] = class_map[nodeid]
            if G.node[nodeid]['test']:
                self.test_idx.append(id_map[nodeid])
            elif G.node[nodeid]['val']:
                self.val_idx.append(id_map[nodeid])
            else:
                self.train_idx.append(id_map[nodeid])
        self.X = np.array(feats, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.long)
        self.transforms = transforms

    def __getitem__(self, index):
        single_label = [self.labels[index][0]]
        single_X = self.X[index]
        if self.transforms is not None:
            single_X = self.transforms(single_X)
        return (single_X, single_label)

    def __len__(self):
        return self.num_samples


class FCNet(nn.Module):
    """Fully connected neural network with ReLU activation function."""
    def __init__(self, 
                 num_input=50,
                 hiddens=[100], 
                 num_labels=121):
        super(FCNet, self).__init__()
        self.fc_layers = [nn.Linear(num_input, hiddens[0])]
        for i in range(1, len(hiddens)):
            self.fc_layers.append(nn.Linear(hiddens[i-1], hiddens[i]))

    def forward(self, x):
        for layers in self.hiddens[:-1]:
            x = F.relu(layers(x))
        last_layer = self.hiddens[-1]
        return last_layer(x)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.BCEWithLogitsLoss()(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item() 
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PPI Feature Only')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--shuffle', action='store_true', default=False)
    parser.add_argument('--train-prefix', type=str, help='point to training')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    dataset = FeaturesOnlyDataset(args.train_prefix)

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    if args.shuffle:
        np.random.seed(args.seed)
        np.random.shuffle(dataset.train_idx)
        np.random.shuffle(dataset.test_idx)
        np.random.shuffle(dataset.val_idx)
    train_indices, val_indices = dataset.train_idx, dataset.val_idx
    test_indices = dataset.test_idx
    
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    train_loader = torch.utils.data.DataLoader(dataset, 
                                               batch_size=args.batch_size, 
                                               sampler=train_sampler)

    validation_loader = torch.utils.data.DataLoader(dataset, 
                                                    batch_size=args.batch_size,
                                                    sampler=valid_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, 
                                              batch_size=args.test_batch_size,
                                              sampler=test_sampler)

    criterion = nn.CrossEntropyLoss()

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), 
                          lr=args.lr, 
                          momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(),"feats_2layers.pt")
        
if __name__ == '__main__':
    main()
