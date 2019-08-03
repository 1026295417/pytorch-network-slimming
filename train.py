import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import cifar

from networks import resnet18, vgg11, vgg11s, densenet63
from netslim import prune, update_bn, update_bn_by_names, get_norm_layer_names, liu2017_normalized_by_layer

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

archs = {
    "resnet18": resnet18, 
    "vgg11": vgg11, "vgg11s": vgg11s, 
    "densenet63": densenet63
}

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Cifar-100 Example for Network Slimming')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=50, metavar='N',
                    help='input batch size for testing (default: 50)')
parser.add_argument('--resume-path', default='',
                    help='path to a trained model weight')
parser.add_argument('--arch', default='resnet18',
                    help='network architecture')
parser.add_argument('--epochs', type=int, default=220, metavar='EP',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='base learning rate (default: 0.1)')
parser.add_argument('--lr-decay-epochs', type=int, default=50, metavar='LR-T',
                    help='the period of epochs to decay LR')
parser.add_argument('--lr-decay-factor', type=float, default=0.3162, metavar='LR-MUL',
                    help='decay factor of learning rate (default: 0.3162)')
parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='L2',
                    help='learning rate (default: 5e-4)')
parser.add_argument('--l1-decay', type=float, default=0, metavar='L1',
                    help='coefficient for L1 regularization on BN (default: 0)')
parser.add_argument('--prune-ratio', type=float, default=-1, metavar='PR',
                    help='ratio of pruned channels to total channels, -1: do not prune')
parser.add_argument('--all-bn', action='store_true', default=False,
                    help='L1 regularization on all BNs, otherwise only on prunable BN')
parser.add_argument('--momentum', type=float, default=0.85, metavar='M',
                    help='SGD momentum (default: 0.85)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='LOG-T',
                    help='how many batches to wait before logging training status')
parser.add_argument('--outf', default='output-cifar-100', metavar='OUTNAME', 
                    help='folder to output images and model checkpoints')
parser.add_argument('--experimental', action='store_true', default=False,
                    help='Normalize scaling factor per layer for pruning')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device('cuda' if args.cuda else 'cpu')

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
normalize = transforms.Normalize(mean=[0.4914, 0.482, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])

train_loader = torch.utils.data.DataLoader(
    cifar.CIFAR100('./cifar-100', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.RandomCrop(32, padding=4),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       normalize
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    cifar.CIFAR100('./cifar-100', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       normalize
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

model = archs[args.arch](num_classes=100)
bn_names = []
if args.l1_decay > 0:
    if not args.all_bn:
        bn_names = get_norm_layer_names(model, (3, 32, 32))
        print("Sparsity regularization will be applied to:")
        for bn_name in bn_names:
            print(bn_name)
    else:
        print("Sparsity regularization will be applied to all BN layers:")
        
model = model.to(device)

if args.prune_ratio > 0 and args.resume_path:
    model.load_state_dict(torch.load(args.resume_path))
    if args.experimental:
        model = prune(model, (3, 32, 32), args.prune_ratio, prune_method=liu2017_normalized_by_layer)
    else:
        model = prune(model, (3, 32, 32), args.prune_ratio)
        
lsm = nn.LogSoftmax(dim=1)
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(lsm(output), target)
        loss.backward()
        if args.all_bn:
            update_bn(model, args.l1_decay)
        else:
            update_bn_by_names(model, bn_names, args.l1_decay)
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch+1, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(lsm(output), target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1]   # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * float(correct) / float(len(test_loader.dataset))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))

    with open('{}.log'.format(args.outf), 'a') as f:
        f.write('{}\t{}\n'.format(epoch, accuracy))

    return accuracy


max_accuracy = 0.
os.system('mkdir -p {}'.format(args.outf))

lr = args.lr
for epoch in range(args.epochs):
    if epoch > 0 and epoch % args.lr_decay_epochs == 0:
        lr *= args.lr_decay_factor
        print('Changing learning rate to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    train(epoch)
    accuracy = test(epoch)
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        torch.save(model.state_dict(), '{}/ckpt_best.pth'.format(args.outf))
    torch.save(model.state_dict(), '{}/ckpt_last.pth'.format(args.outf))
    