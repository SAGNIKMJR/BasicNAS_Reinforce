import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# from lib.Datasets.preprocessing import Preprocessing
import preprocessing as Preprocessing
import math
import os
import struct
import gzip
import errno
import numpy as np


class CUSTOM:
    """
    CUSTOM dataset using ImageFolder for any data-loading.
    Preprocessing is calculated using the preprocessing class.

    Parameters:
        args (dict): Dictionary of (command line) arguments.
            Needs to contain batch_size (int), workers(int),
            patch_size (int) defining image size and
            two paths train_data (str) and val_data (str)
        is_gpu (bool): True if CUDA is enabled.
            Sets value of pin_memory in DataLoader.

    Attributes:
        traindir (str): Path to train dataset as required by
            torchvision.datasets.ImageFolder
        valdir (str): Path to validation dataset as required by
            torchvision.datasets.ImageFolder
        normalize (dict): Contains per-channel means and stds of the dataset.
        train_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, horizontal flips, random
            translations up to 10% of the image in each direction and
            normalization.
        val_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, scaling to patch size
            and normalization.
        trainset (torch.utils.data.TensorDataset): Training set wrapper.
        valset (torch.utils.data.TensorDataset): Validation set wrapper.
        train_loader (torch.utils.data.DataLoader): Training set loader with shuffling.
        val_loader (torch.utils.data.DataLoader): Validation set loader.
    """

    def __init__(self, is_gpu, args):
        self.valdir = args.val_data
        self.traindir = args.train_data

        self.normalize = self.__get_normalize(args.workers)
        self.train_transforms, self.val_transforms = self.__get_transforms(args.patch_size)

        self.trainset, self.valset = self.get_dataset()
        self.train_loader, self.val_loader = self.get_dataset_loader(args.batch_size, args.workers, is_gpu)

    def __get_normalize(self, workers):
        preprocess_params = Preprocessing()
        preprocess_params.get_mean_and_std(self.traindir, workers)
        normalize = transforms.Normalize(preprocess_params.mean, preprocess_params.std)
        return normalize

    def __get_transforms(self, patch_size):
    #images are assumed to be square images or they will be warped
        train_transforms = transforms.Compose([
            transforms.Scale(size=(patch_size, patch_size)),
            transforms.RandomCrop(patch_size, int(math.ceil(patch_size * 0.1))),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize,
            ])

        val_transforms = transforms.Compose([
            transforms.Scale(size=(patch_size, patch_size)),
            transforms.ToTensor(),
            self.normalize,
            ])

        return train_transforms, val_transforms

    def get_dataset(self):
        """
        Uses torchvision.datasets.ImageFolder to load dataset
        from traindir and valdir respectively.

        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        trainset = datasets.ImageFolder(self.traindir, self.train_transforms)
        valset = datasets.ImageFolder(self.valdir, self.val_transforms)

        return trainset, valset

    def get_dataset_loader(self, batch_size, workers, is_gpu):
        """
        Defines the dataset loader for wrapped dataset

        Parameters:
            batch_size (int): Defines the batch size in data loader
            workers (int): Number of parallel threads to be used by data loader
            is_gpu (bool): True if CUDA is enabled so pin_memory is set to True

        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)

        return train_loader, val_loader


class IMAGENET:
    """
    Actually just a variant of CUSTOM dataset and uses ImageFolder
    for any data-loading. Main difference is in definition of transforms
    and pre-calculated preprocessing parameters for 1000 class ILSVRC
    2012 dataset.

    Parameters:
        args (dict): Dictionary of (command line) arguments.
            Needs to contain batch_size (int), workers(int)
            and two paths train_data (str) and val_data (str)
        is_gpu (bool): True if CUDA is enabled.
            Sets value of pin_memory in DataLoader.

    Attributes:
        traindir (str): Path to train dataset as required by
            torchvision.datasets.ImageFolder
        valdir (str): Path to validation dataset as required by
            torchvision.datasets.ImageFolder
        normalize (dict): Contains per-channel means and stds of the dataset.
        train_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, horizontal flips, random
            crops of size 224 x 224 and normalization.
        val_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, scaling of shorter side to
            256 followed by a center crop of 224 x 224 and normalization.
        trainset (torch.utils.data.TensorDataset): Training set wrapper.
        valset (torch.utils.data.TensorDataset): Validation set wrapper.
        train_loader (torch.utils.data.DataLoader): Training set loader with shuffling.
        val_loader (torch.utils.data.DataLoader): Validation set loader.
    """

    def __init__(self, is_gpu, args):
        self.valdir = args.val_data
        self.traindir = args.train_data

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.train_transforms, self.val_transforms = self.__get_transforms()

        self.trainset, self.valset = self.get_dataset()
        self.train_loader, self.val_loader = self.get_dataset_loader(args.batch_size, args.workers, is_gpu)

    def __get_transforms(self):
        train_transforms = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize,
            ])

        val_transforms = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            self.normalize,
            ])

        return train_transforms, val_transforms

    def get_dataset(self):
        """
        Uses torchvision.datasets.ImageFolder to load dataset
        from traindir and valdir respectively.

        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """
        trainset = datasets.ImageFolder(self.traindir, self.train_transforms)
        valset = datasets.ImageFolder(self.valdir, self.val_transforms)

        return trainset, valset

    def get_dataset_loader(self, batch_size, workers, is_gpu):
        """
        Defines the dataset loader for wrapped dataset

        Parameters:
            batch_size (int): Defines the batch size in data loader
            workers (int): Number of parallel threads to be used by data loader
            is_gpu (bool): True if CUDA is enabled so pin_memory is set to True

        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)

        return train_loader, val_loader


class CIFAR10:
    """
    CIFAR-10 dataset featuring tiny 32x32 color images of
    objects belonging to hundred different classes.
    Dataloader adapted from CIFAR10.

    Parameters:
        args (dict): Dictionary of (command line) arguments.
            Needs to contain batch_size (int) and workers(int).
        is_gpu (bool): True if CUDA is enabled.
            Sets value of pin_memory in DataLoader.

    Attributes:
        normalize (dict): Contains per-channel means and stds of the dataset.
        train_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, horizontal flips, random
            translations of up to 10% in each direction and normalization.
        val_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor and normalization.
        trainset (torch.utils.data.TensorDataset): Training set wrapper.
        valset (torch.utils.data.TensorDataset): Validation set wrapper.
        train_loader (torch.utils.data.DataLoader): Training set loader with shuffling
        val_loader (torch.utils.data.DataLoader): Validation set loader.
    """

    def __init__(self, is_gpu, args):
        self.normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                              std=[0.2023, 0.1994, 0.2010])

        self.train_transforms, self.val_transforms = self.__get_transforms(args.patch_size)

        self.trainset, self.valset = self.get_dataset()
        self.train_loader, self.val_loader = self.get_dataset_loader(args.batch_size, args.workers, is_gpu)

        # Need to define the class dictionary by hand as the default
        # torchvision CIFAR10 data loader does not provide class_to_idx
        self.val_loader.dataset.class_to_idx = {'airplane': 0,
                                                'automobile': 1,
                                                'bird': 2,
                                                'cat': 3,
                                                'deer': 4,
                                                'dog': 5,
                                                'frog': 6,
                                                'horse': 7,
                                                'ship': 8,
                                                'truck': 9}

    def __get_transforms(self, patch_size):
        train_transforms = transforms.Compose([
            transforms.RandomCrop(patch_size, int(math.ceil(patch_size * 0.1))),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize,
            ])

        val_transforms = transforms.Compose([
            transforms.Scale(patch_size),
            transforms.ToTensor(),
            self.normalize,
            ])

        return train_transforms, val_transforms

    def get_dataset(self):
        """
        Uses torchvision.datasets.CIFAR10 to load dataset.
        Downloads dataset if doesn't exist already.

        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        trainset = datasets.CIFAR10('datasets/CIFAR10/train/', train=True, transform=self.train_transforms,
                                    target_transform=None, download=True)
        valset = datasets.CIFAR10('datasets/CIFAR10/test/', train=False, transform=self.val_transforms,
                                  target_transform=None, download=True)

        return trainset, valset

    def get_dataset_loader(self, batch_size, workers, is_gpu):
        """
        Defines the dataset loader for wrapped dataset

        Parameters:
            batch_size (int): Defines the batch size in data loader
            workers (int): Number of parallel threads to be used by data loader
            is_gpu (bool): True if CUDA is enabled so pin_memory is set to True

        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)

        return train_loader, val_loader


class CIFAR100:
    """
    CIFAR-100 dataset featuring tiny 32x32 color images of
    objects belonging to hundred different classes.
    Dataloader adapted from CIFAR10.

    Parameters:
        args (dict): Dictionary of (command line) arguments.
            Needs to contain batch_size (int) and workers(int).
        is_gpu (bool): True if CUDA is enabled.
            Sets value of pin_memory in DataLoader.

    Attributes:
        normalize (dict): Contains per-channel means and stds of the dataset.
        train_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, horizontal flips, random
            translations of up to 10% in each direction and normalization.
        val_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor and normalization.
        trainset (torch.utils.data.TensorDataset): Training set wrapper.
        valset (torch.utils.data.TensorDataset): Validation set wrapper.
        train_loader (torch.utils.data.DataLoader): Training set loader with shuffling.
        val_loader (torch.utils.data.DataLoader): Validation set loader.
    """

    def __init__(self, is_gpu, args):
        self.normalize = transforms.Normalize(mean=[0.5071, 0.4866, 0.4409],
                                              std=[0.2009, 0.1984, 0.2023])

        self.train_transforms, self.val_transforms = self.__get_transforms(args.patch_size)

        self.trainset, self.valset = self.get_dataset()
        self.train_loader, self.val_loader = self.get_dataset_loader(args.batch_size, args.workers, is_gpu)

        # Need to define the class dictionary by hand as the default
        # torchvision CIFAR100 data loader does not provide class_to_idx
        self.val_loader.dataset.class_to_idx = {'apples': 0,
                                                'aquariumfish': 1,
                                                'baby': 2,
                                                'bear': 3,
                                                'beaver': 4,
                                                'bed': 5,
                                                'bee': 6,
                                                'beetle': 7,
                                                'bicycle': 8,
                                                'bottles': 9,
                                                'bowls': 10,
                                                'boy': 11,
                                                'bridge': 12,
                                                'bus': 13,
                                                'butterfly': 14,
                                                'camel': 15,
                                                'cans': 16,
                                                'castle': 17,
                                                'caterpillar': 18,
                                                'cattle': 19,
                                                'chair': 20,
                                                'chimpanzee': 21,
                                                'clock': 22,
                                                'cloud': 23,
                                                'cockroach': 24,
                                                'computerkeyboard': 25,
                                                'couch': 26,
                                                'crab': 27,
                                                'crocodile': 28,
                                                'cups': 29,
                                                'dinosaur': 30,
                                                'dolphin': 31,
                                                'elephant': 32,
                                                'flatfish': 33,
                                                'forest': 34,
                                                'fox': 35,
                                                'girl': 36,
                                                'hamster': 37,
                                                'house': 38,
                                                'kangaroo': 39,
                                                'lamp': 40,
                                                'lawnmower': 41,
                                                'leopard': 42,
                                                'lion': 43,
                                                'lizard': 44,
                                                'lobster': 45,
                                                'man': 46,
                                                'maple': 47,
                                                'motorcycle': 48,
                                                'mountain': 49,
                                                'mouse': 50,
                                                'mushrooms': 51,
                                                'oak': 52,
                                                'oranges': 53,
                                                'orchids': 54,
                                                'otter': 55,
                                                'palm': 56,
                                                'pears': 57,
                                                'pickuptruck': 58,
                                                'pine': 59,
                                                'plain': 60,
                                                'plates': 61,
                                                'poppies': 62,
                                                'porcupine': 63,
                                                'possum': 64,
                                                'rabbit': 65,
                                                'raccoon': 66,
                                                'ray': 67,
                                                'road': 68,
                                                'rocket': 69,
                                                'roses': 70,
                                                'sea': 71,
                                                'seal': 72,
                                                'shark': 73,
                                                'shrew': 74,
                                                'skunk': 75,
                                                'skyscraper': 76,
                                                'snail': 77,
                                                'snake': 78,
                                                'spider': 79,
                                                'squirrel': 80,
                                                'streetcar': 81,
                                                'sunflowers': 82,
                                                'sweetpeppers': 83,
                                                'table': 84,
                                                'tank': 85,
                                                'telephone': 86,
                                                'television': 87,
                                                'tiger': 88,
                                                'tractor': 89,
                                                'train': 90,
                                                'trout': 91,
                                                'tulips': 92,
                                                'turtle': 93,
                                                'wardrobe': 94,
                                                'whale': 95,
                                                'willow': 96,
                                                'wolf': 97,
                                                'woman': 98,
                                                'worm': 99}

    def __get_transforms(self, patch_size):
        train_transforms = transforms.Compose([
            transforms.RandomCrop(patch_size, int(math.ceil(patch_size * 0.1))),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize,
            ])

        val_transforms = transforms.Compose([
            transforms.Scale(patch_size),
            transforms.ToTensor(),
            self.normalize,
            ])

        return train_transforms, val_transforms

    def get_dataset(self):
        """
        Uses torchvision.datasets.CIFAR100 to load dataset.
        Downloads dataset if doesn't exist already.

        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        trainset = datasets.CIFAR100('datasets/CIFAR100/train/', train=True, transform=self.train_transforms,
                                     target_transform=None, download=True)
        valset = datasets.CIFAR100('datasets/CIFAR100/test/', train=False, transform=self.val_transforms,
                                   target_transform=None, download=True)

        return trainset, valset

    def get_dataset_loader(self, batch_size, workers, is_gpu):
        """
        Defines the dataset loader for wrapped dataset

        Parameters:
            batch_size (int): Defines the batch size in data loader
            workers (int): Number of parallel threads to be used by data loader
            is_gpu (bool): True if CUDA is enabled so pin_memory is set to True

        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)

        return train_loader, val_loader


class MNIST:
    """
    MNIST dataset featuring gray-scale 28x28 images of
    hand-written characters belonging to ten different classes.
    Dataset implemented with torchvision.datasets.MNIST.

    Parameters:
        args (dict): Dictionary of (command line) arguments.
            Needs to contain batch_size (int) and workers(int).
        is_gpu (bool): True if CUDA is enabled.
            Sets value of pin_memory in DataLoader.

    Attributes:
        normalize (dict): Contains per-channel means and stds of the dataset.
        train_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, repeating gray-scale image to
            three channel for consistent use with different architectures
            and normalization.
        val_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, repeating gray-scale image to
            three channel for consistent use with different architectures
            and normalization.
        trainset (torch.utils.data.TensorDataset): Training set wrapper.
        valset (torch.utils.data.TensorDataset): Validation set wrapper.
        train_loader (torch.utils.data.DataLoader): Training set loader with shuffling.
        val_loader (torch.utils.data.DataLoader): Validation set loader.
    """

    def __init__(self, is_gpu, patch_size, train_batch_size, val_batch_size, workers, train_dir, val_dir):
        self.normalize = transforms.Normalize(mean=[0.1299, 0.1299, 0.1299],
                                              std=[0.3002, 0.3002, 0.3002])

        self.traindir = train_dir
        self.valdir = val_dir

        self.train_transforms, self.val_transforms = self.__get_transforms(patch_size)

        self.trainset, self.valset = self.get_dataset()
        self.train_loader, self.val_loader = self.get_dataset_loader(train_batch_size, val_batch_size, workers, is_gpu)

        # Need to define the class dictionary by hand as the default
        # torchvision MNIST data loader does not provide class_to_idx
        self.val_loader.dataset.class_to_idx = {'0': 0,
                                                '1': 1,
                                                '2': 2,
                                                '3': 3,
                                                '4': 4,
                                                '5': 5,
                                                '6': 6,
                                                '7': 7,
                                                '8': 8,
                                                '9': 9}

    def __get_transforms(self, patch_size):
        # scale the images (e.g. to 32x32, so the same model
        # as for CIFAR10 can be used for comparison
        # for analogous reasons we also define a lambda transform
        # to duplicate the gray-scale image to 3 channels
        train_transforms = transforms.Compose([
            transforms.Scale(patch_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            self.normalize,
            ])

        val_transforms = transforms.Compose([
            transforms.Scale(patch_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            self.normalize,
            ])

        return train_transforms, val_transforms

    def get_dataset(self):
        """
        Uses torchvision.datasets.MNIST to load dataset.
        Downloads dataset if doesn't exist already.

        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        trainset = datasets.MNIST(self.traindir, train=True, transform=self.train_transforms,
                                  target_transform=None, download=True)
        valset = datasets.MNIST(self.valdir, train=False, transform=self.val_transforms,
                                target_transform=None, download=True)

        # trainset = datasets.ImageFolder(self.traindir, self.train_transforms)
        # valset = datasets.ImageFolder(self.valdir, self.val_transforms)

        return trainset, valset

    def get_dataset_loader(self, train_batch_size, val_batch_size,workers, is_gpu):
        """
        Defines the dataset loader for wrapped dataset

        Parameters:
            batch_size (int): Defines the batch size in data loader
            workers (int): Number of parallel threads to be used by data loader
            is_gpu (bool): True if CUDA is enabled so pin_memory is set to True

        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=train_batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=val_batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)

        return train_loader, val_loader


class FashionMNIST:
    """
    Fashion MNIST dataset featuring gray-scale 28x28 images of
    fashion items belonging to ten different classes.
    Dataloader adapted from MNIST.
    We do not define __getitem__ and __len__ in this class
    as we are using torch.utils.data.TensorDataSet which
    already implements these methods.

    Parameters:
        args (dict): Dictionary of (command line) arguments.
            Needs to contain batch_size (int) and workers(int).
        is_gpu (bool): True if CUDA is enabled.
            Sets value of pin_memory in DataLoader.

    Attributes:
        trainset (torch.utils.data.TensorDataset): Training set wrapper.
        valset (torch.utils.data.TensorDataset): Validation set wrapper.
        train_loader (torch.utils.data.DataLoader): Training set loader with shuffling.
        val_loader (torch.utils.data.DataLoader): Validation set loader.
    """

    def __init__(self, is_gpu, args):
        self.__path = os.path.expanduser('datasets/FashionMNIST')
        self.__download()

        self.trainset, self.valset = self.get_dataset()
        self.train_loader, self.val_loader = self.get_dataset_loader(args.batch_size, args.workers, is_gpu)

        self.val_loader.dataset.class_to_idx = {'T-shirt/top': 0,
                                                'Trouser': 1,
                                                'Pullover': 2,
                                                'Dress': 3,
                                                'Coat': 4,
                                                'Sandal': 5,
                                                'Shirt': 6,
                                                'Sneaker': 7,
                                                'Bag': 8,
                                                'Ankle boot': 9}

    def __check_exists(self):
        """
        Checks if dataset has already been downloaded

        Returns:
             bool: True if downloaded dataset has been found
        """
        return os.path.exists(os.path.join(self.__path, 'train-images-idx3-ubyte.gz')) and \
               os.path.exists(os.path.join(self.__path, 'train-labels-idx1-ubyte.gz')) and \
               os.path.exists(os.path.join(self.__path, 't10k-images-idx3-ubyte.gz')) and \
               os.path.exists(os.path.join(self.__path, 't10k-labels-idx1-ubyte.gz'))

    def __download(self):
        """
        Downloads the Fashion-MNIST dataset from the web if dataset
        hasn't already been downloaded.
        """
        from six.moves import urllib

        if self.__check_exists():
            return

        print("Downloading FashionMNIST dataset")
        urls = [
            'https://cdn.rawgit.com/zalandoresearch/fashion-mnist/ed8e4f3b/data/fashion/train-images-idx3-ubyte.gz',
            'https://cdn.rawgit.com/zalandoresearch/fashion-mnist/ed8e4f3b/data/fashion/train-labels-idx1-ubyte.gz',
            'https://cdn.rawgit.com/zalandoresearch/fashion-mnist/ed8e4f3b/data/fashion/t10k-images-idx3-ubyte.gz',
            'https://cdn.rawgit.com/zalandoresearch/fashion-mnist/ed8e4f3b/data/fashion/t10k-labels-idx1-ubyte.gz',
        ]

        # download files
        try:
            os.makedirs(self.__path)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.__path, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())

        print('Done!')

    def __get_fashion_mnist(self, path, kind='train'):
        """
        Load Fashion-MNIST data

        Parameters:
            path (str): Base directory path containing .gz files for
                the Fashion-MNIST dataset
            kind (str): Accepted types are 'train' and 't10k' for
                training and validation set stored in .gz files

        Returns:
            numpy.array: images, labels

        """
        labels_path = os.path.join(path,
                                   '%s-labels-idx1-ubyte.gz'
                                   % kind)
        images_path = os.path.join(path,
                                   '%s-images-idx3-ubyte.gz'
                                   % kind)

        with gzip.open(labels_path, 'rb') as lbpath:
            struct.unpack('>II', lbpath.read(8))
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8)

        with gzip.open(images_path, 'rb') as imgpath:
            struct.unpack(">IIII", imgpath.read(16))
            images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), 784)

        return images, labels

    def get_dataset(self):
        """
        Loads and wraps training and validation datasets

        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """
        x_train, y_train = self.__get_fashion_mnist(self.__path, kind='train')
        x_val, y_val = self.__get_fashion_mnist(self.__path, kind='t10k')

        # convert to torch tensors in range [0, 1]
        x_train = torch.from_numpy(x_train).float() / 255
        y_train = torch.from_numpy(y_train).int()
        x_val = torch.from_numpy(x_val).float() / 255
        y_val = torch.from_numpy(y_val).int()

        # resize flattened array of images for input to a CNN
        x_train.resize_(x_train.size(0), 1, 28, 28)
        x_val.resize_(x_val.size(0), 1, 28, 28)
        # PyTorch always expects three channels, even for gray-scale images
        x_train = x_train.repeat(1, 3, 1, 1)
        x_val = x_val.repeat(1, 3, 1, 1)

        # TensorDataset wrapper
        trainset = torch.utils.data.TensorDataset(x_train, y_train)
        valset = torch.utils.data.TensorDataset(x_val, y_val)

        return trainset, valset

    def get_dataset_loader(self, batch_size, workers, is_gpu):
        """
        Defines the dataset loader for wrapped dataset

        Parameters:
            batch_size (int): Defines the batch size in data loader
            workers (int): Number of parallel threads to be used by data loader
            is_gpu (bool): True if CUDA is enabled so pin_memory is set to True

        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """
        train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True,
                                                   num_workers=workers, pin_memory=is_gpu, sampler=None)
        test_loader = torch.utils.data.DataLoader(self.valset, batch_size=batch_size, shuffle=False,
                                                  num_workers=workers, pin_memory=is_gpu, sampler=None)

        return train_loader, test_loader
