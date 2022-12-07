import argparse
import torch
import os

import torchvision.transforms as transforms

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from utils.utils import print_network
from models.resnet_custom import resnet50_baseline
from utils.core_utils import EarlyStopping

parser = argparse.ArgumentParser(description='Finetuning Feature Extractor')
parser.add_argument('--image_dir', type=str, default=None)
parser.add_argument('--result_dir', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--model_path', type=str, default=None)

parser.add_argument('--n_epochs', type=int, default=100,
                    help='number of epochs')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    args = parser.parse_args()

    # initalize dataset
    normalize = transforms.Normalize(mean=[0.7785, 0.6139, 0.7132],
                                     std=[0.1942, 0.2412, 0.1882])
    augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.1) 
            ], p=0.7),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            normalize
        ]
    transform = transforms.Compose(augmentation)
    dataset = ImageFolder(args.image_dir, transform=transform)

    # train / val split with 80% training images
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    kwargs = {'num_workers': 4} if device.type == "cuda" else {}

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, **kwargs)

    # initialize feature extractor
    print('loading model checkpoint')
    resnet = resnet50_baseline(pretrained=True, model_path=args.model_path)

    # freeze ResNet except of last layer
    for name, params in resnet.named_parameters():
        if not 'layer3' in name:
            params.requires_grad = False

    # add classification layers to network
    fc = [nn.Dropout(p=0.3), 
        nn.Linear(1024, 2)]
    model = nn.Sequential(resnet, *fc)
    print_network(model)

    # unfreeze layers and set specific learning rates
    model = model.to(device)

    # initialize loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)

    early_stopping = EarlyStopping(patience = 20, stop_epoch=100, verbose = True)

    for epoch in range(args.n_epochs):

        train_loss = 0
        train_acc = 0

        model.train()
        # train
        for batch_idx, (data, label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device)

            pred = model(data)
            Y_prob = nn.functional.softmax(pred, dim = 1)
            loss = loss_fn(Y_prob, label)
            train_loss += loss.item()
            acc = (torch.argmax(Y_prob, dim=1) == label).float().sum() / args.batch_size
            train_acc += acc

            loss.backward()
            # step
            optimizer.step()
            optimizer.zero_grad()

            if (batch_idx + 1) % 20 == 0:
                print('epoch [{}], batch {}, loss: {:.4f}, accuracy: {:.4f}'.format(epoch, batch_idx, loss.item(), acc))

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        print('train stats:')
        print('epoch [{}], loss: {:.4f}, accuracy: {:.4f}'.format(epoch, train_loss, train_acc))

        # validate
        val_loss = 0
        val_acc = 0

        model.eval()
        for batch_idx, (data, label) in enumerate(val_loader):
            data, label = data.to(device), label.to(device)

            pred = model(data)
            Y_prob = nn.functional.softmax(pred, dim = 1)
            val_loss += loss_fn(Y_prob, label).item()
            val_acc += (torch.argmax(Y_prob, dim=1) == label).float().sum() / args.batch_size

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        print('val stats:')
        print('epoch [{}], loss: {:.4f}, accuracy: {:.4f}'.format(epoch, val_loss, val_acc))

        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(args.result_dir, "checkpoint_epoch_{}.pt".format(epoch)))

        if early_stopping.early_stop:
            print("Early stopping")
            break


if __name__ == '__main__':
    main()