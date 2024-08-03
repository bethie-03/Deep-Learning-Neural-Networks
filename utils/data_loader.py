import numpy as np
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

def data_loader(data_dir, batch_size, random_seed = 42, shuffle = True, valid_size=0.1, test = False):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = (0.4914, 0.4822, 0.4465),
            std = (0.2023, 0.1994, 0.2010)
        )
    ])

    if test:
        dataset = datasets.CIFAR100(
            root = data_dir,
            train = False,
            download = True,
            transform = transform
        )

        data_loader = DataLoader(
            dataset, batch_size = batch_size, shuffle = shuffle
        )

        return data_loader

    train_dataset = datasets.CIFAR100(
        root = data_dir,
        train = True,
        download = True,
        transform = transform
    )

    '''valid_dataset = datasets.CIFAR10(
        root = data_dir,
        train = True,
        download = True,
        transform = transform
    )'''

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size*num_train))

    if shuffle:
      np.random.seed(random_seed)
      np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(
       train_dataset, batch_size=batch_size, sampler=train_sampler
    )

    valid_loader = DataLoader(
       train_dataset, batch_size=batch_size, sampler=valid_sampler
    )

    return (train_loader, valid_loader)

