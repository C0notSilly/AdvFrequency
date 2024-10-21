import torch
import torch.utils.data as data

from PIL import Image
import os
import numpy as np

from utils.AugMixDataset import AugMixDataset

from torchvision.datasets import ImageFolder
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import warnings


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive


class MNISTM(VisionDataset):
    """MNIST-M Dataset.
    """

    resources = [
        ('https://github.com/liyxi/mnist-m/releases/download/data/mnist_m_train.pt.tar.gz',
         '191ed53db9933bd85cc9700558847391'),
        ('https://github.com/liyxi/mnist-m/releases/download/data/mnist_m_test.pt.tar.gz',
         'e11cb4d7fff76d7ec588b1134907db59')
    ]

    training_file = "mnist_m_train.pt"
    test_file = "mnist_m_test.pt"
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        """Init MNIST-M dataset."""
        super(MNISTM, self).__init__(root, transform=transform, target_transform=target_transform)

        self.train = train

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        print(os.path.join(self.processed_folder, data_file))

        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.squeeze().numpy(), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Return size of dataset."""
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder, self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder, self.test_file)))

    def download(self):
        """Download the MNIST-M data."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder,
                                         extract_root=self.processed_folder,
                                         filename=filename, md5=md5)

        print('Done!')

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


class SyntheticDigits(VisionDataset):
    """Synthetic Digits Dataset.
    """

    resources = [
        ('https://github.com/liyxi/synthetic-digits/releases/download/data/synth_train.pt.gz',
         'd0e99daf379597e57448a89fc37ae5cf'),
        ('https://github.com/liyxi/synthetic-digits/releases/download/data/synth_test.pt.gz',
         '669d94c04d1c91552103e9aded0ee625')
    ]

    training_file = "synth_train.pt"
    test_file = "synth_test.pt"
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        """Init Synthetic Digits dataset."""
        super(SyntheticDigits, self).__init__(root, transform=transform, target_transform=target_transform)

        self.train = train

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        print(os.path.join(self.processed_folder, data_file))

        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.squeeze().numpy(), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Return size of dataset."""
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder, self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder, self.test_file)))

    def download(self):
        """Download the Synthetic Digits data."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder,
                                         extract_root=self.processed_folder,
                                         filename=filename, md5=md5)

        print('Done!')

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')


def make_dataset(root, label):
    images = []
    labeltxt = open(label)
    for line in labeltxt:
        data = line.strip().split(' ')
        if is_image_file(data[0]):
            path = os.path.join(root, data[0])
        gt = int(data[1])
        item = (path, gt)
        images.append(item)
    return images


class OfficeImage(data.Dataset):
    def __init__(self, root, label, transform=None, loader=default_loader, do_process=True):
        imgs = make_dataset(root, label)
        self.root = root
        self.label = label
        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        self.do_process = do_process

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)

        if self.transform is not None and self.do_process is True:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)


def LoadCIFAR_Train(root, batch_size, transform):
    train_set = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8,
                                               pin_memory=4, worker_init_fn=_init_fn)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=8)
    return train_loader, test_loader


def LoadMNIST_Train(root, batch_size, transform):
    train_set = datasets.MNIST(root+'/MNIST', train=True, transform=transform)
    test_trans = transforms.Compose([
        transforms.Resize(36),
        transforms.CenterCrop(32),
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor()
    ])
    test_set = datasets.MNIST(root+'/MNIST', train=False, transform=test_trans)
    # train_aug_set = AugMixDataset(train_set, transforms.ToTensor(), image_size=32)
    train_aug_set = AugMixDataset(train_set, test_trans, image_size=32)
    train_loader = torch.utils.data.DataLoader(train_aug_set, batch_size=batch_size, shuffle=True, num_workers=0,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=0,
                                               pin_memory=True)
    return train_loader, test_loader

def LoadMNIST_Test(root, batch_size, transform):
    domain_list = ['MNISTM', 'SVHN', 'USPS', 'SYNTH']
    loader_dict = {}

    svhn_data = datasets.SVHN(root+'/SVHN/', split='test', download=True, transform=transform)
    svhn_loader = torch.utils.data.DataLoader(svhn_data, batch_size=batch_size, num_workers=8)
    loader_dict.update({'SVHN':svhn_loader})

    mnistm = MNISTM(root=root+'/MNISTM/', train=False, transform=transform, download=True)
    mnistm_loader = torch.utils.data.DataLoader(mnistm, batch_size=batch_size, num_workers=8)
    loader_dict.update({'MNISTM':mnistm_loader})

    synth = SyntheticDigits(root=root+'/SYNTH/', train=False, transform=transform, download=True)
    synth_loader = torch.utils.data.DataLoader(synth, batch_size=batch_size, num_workers=8)
    loader_dict.update({"SYNTH":synth_loader})

    usps_set = ImageFolder(root+'/USPS/torch', transform=transform)
    usps_loader = torch.utils.data.DataLoader(usps_set, batch_size=batch_size, num_workers=8)
    loader_dict.update({"USPS":usps_loader})

    return loader_dict


def LoadDataset_SingleSource(dataset_name, source, root, batch_size, mode, transform=None,
                             loader=default_loader):
    if dataset_name == 'PACS':
        if mode == 'train':
            return LoadPACS_Train(source, root, batch_size, transform)
        elif mode == 'test':
            return LoadPACS_Test(source, root, batch_size, transform)
        elif mode == 'train_augmix':
            return LoadPACS_AugMix(source, root, batch_size, transform)
    elif dataset_name == 'CIFAR':
        if mode == 'train':
            return LoadCIFAR_Train(root, batch_size, transform)
        elif mode == 'train_augmix':
            return LoadCIFAR_AugMix(root, batch_size, transform)
    elif dataset_name == 'Digits':
        if mode == 'train_augmix':
            return LoadMNIST_Train(root, batch_size, transform)
        elif mode == 'test':
            return LoadMNIST_Test(root, batch_size, transform)
    elif dataset_name == 'OfficeHome':
        if mode == 'train':
            return LoadOH_Train(source, root, batch_size, transform)
        elif mode == 'test':
            return LoadOH_Test(source, root, batch_size, transform)
        elif mode == 'train_augmix':
            return LoadOH_AugMix(source, root, batch_size, transform)
    else:
        raise ValueError("unincluded dataset !")


seed = 1024


# 设置每个读取线程的随机种子
def _init_fn(worker_id):
    np.random.seed(int(seed) + worker_id)


def LoadPACS_Train(source, root, batch_size, transform):
    source_root = root  # os.path.join(root)
    source_label = os.path.join(root, source + ".txt")
    source_set = OfficeImage(source_root, source_label, transform)
    train_size = int(len(source_set) * 0.9)
    test_size = len(source_set) - train_size
    train_set, val_set = torch.utils.data.random_split(source_set, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=32,
                                               pin_memory=True, worker_init_fn=_init_fn)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, num_workers=32,
                                             pin_memory=True, worker_init_fn=_init_fn)
    return train_loader, val_loader

def LoadOH_Train(source, root, batch_size, transform):
    source_folder = os.path.join(root, source)
    source_set = ImageFolder(source_folder, transform)
    train_size = int(len(source_set) * 0.9)
    test_size = len(source_set) - train_size
    train_set, val_set = torch.utils.data.random_split(source_set, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=32,
                                               pin_memory=True, worker_init_fn=_init_fn)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, num_workers=32,
                                             pin_memory=True, worker_init_fn=_init_fn)
    return train_loader, val_loader


def LoadPACS_Test(source, root, batch_size, transform):
    domain_list = ['photo', 'art_painting', 'cartoon', 'sketch']
    loader_dict = {}
    for index, test_domain in enumerate(domain_list):
        if test_domain == source:
            continue
        test_root = root
        test_label = os.path.join(root, test_domain + ".txt")
        target_set = OfficeImage(test_root, test_label, transform)
        target_loader = torch.utils.data.DataLoader(target_set, batch_size=batch_size, num_workers=32)
        loader_dict.update({test_domain: target_loader})
    return loader_dict


def LoadOH_Test(source, root, batch_size, transform):
    domain_list = ['Art', 'Clipart', 'Product', 'Real World']
    loader_dict = {}
    for index, test_domain in enumerate(domain_list):
        if test_domain == source:
            continue
        test_root = root
        test_domain_folder = os.path.join(root, test_domain)
        target_set = ImageFolder(test_domain_folder, transform)
        target_loader = torch.utils.data.DataLoader(target_set, batch_size=batch_size, num_workers=32)
        loader_dict.update({test_domain: target_loader})
    return loader_dict



def LoadPACS_AugMix(source, root, batch_size, transform):
    init_transform = train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    source_root = root  # os.path.join(root)
    source_label = os.path.join(root, source + ".txt")

    source_aug_set = OfficeImage(source_root, source_label, transform, do_process=False)
    train_size = int(len(source_aug_set) * 0.9)
    test_size = len(source_aug_set) - train_size
    train_aug_set, _ = torch.utils.data.random_split(source_aug_set, [train_size, test_size],
                                                     generator=torch.Generator().manual_seed(42))

    aug_dataset = AugMixDataset(train_aug_set, preprocess=init_transform)

    train_aug_loader = torch.utils.data.DataLoader(aug_dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                                   pin_memory=True, worker_init_fn=_init_fn)

    source_set = OfficeImage(source_root, source_label, transform, do_process=True)

    train_set, val_set = torch.utils.data.random_split(source_set, [train_size, test_size],
                                                       generator=torch.Generator().manual_seed(42))

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, num_workers=8,
                                             pin_memory=True, worker_init_fn=_init_fn)
    return train_aug_loader, val_loader

def LoadCIFAR_AugMix(root, batch_size, transform):
    init_transform = train_transform = transforms.Compose([
        transforms.Resize((36, 36)),
        transforms.CenterCrop((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_set = datasets.CIFAR10(root=root, train=True, download=True, transform=None)
    test_set = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

    aug_train_set = AugMixDataset(train_set, preprocess=init_transform)
    train_aug_loader = torch.utils.data.DataLoader(aug_train_set, batch_size=batch_size, shuffle=True, num_workers=8,
                                                   pin_memory=True, worker_init_fn=_init_fn)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=8,
                                             pin_memory=True, worker_init_fn=_init_fn)
    return train_aug_loader, test_loader

def LoadOH_AugMix(source, root, batch_size, transform):
    init_transform = train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    source_root = root  # os.path.join(root)
    source_folder_path = os.path.join(root, source)

    source_aug_set = ImageFolder(source_folder_path)
    train_size = int(len(source_aug_set) * 0.9)
    test_size = len(source_aug_set) - train_size
    train_aug_set, _ = torch.utils.data.random_split(source_aug_set, [train_size, test_size],
                                                     generator=torch.Generator().manual_seed(42))

    aug_dataset = AugMixDataset(train_aug_set, preprocess=init_transform)

    train_aug_loader = torch.utils.data.DataLoader(aug_dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                                   pin_memory=True, worker_init_fn=_init_fn)

    source_set = ImageFolder(source_folder_path, transform)

    train_set, val_set = torch.utils.data.random_split(source_set, [train_size, test_size],
                                                       generator=torch.Generator().manual_seed(42))

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, num_workers=8,
                                             pin_memory=True, worker_init_fn=_init_fn)
    return train_aug_loader, val_loader
