import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from scipy.stats import wishart
import os

def load_mnist():
    """Load MNIST dataset."""
    print("Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    
    return train_dataset, test_dataset

def load_cifar10():
    """Load CIFAR-10 dataset."""
    print("Loading CIFAR-10 dataset...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    
    return train_dataset, test_dataset

def load_cifar100():
    """Load CIFAR-100 dataset."""
    print("Loading CIFAR-100 dataset...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
    
    return train_dataset, test_dataset

def load_svhn():
    """Load SVHN dataset."""
    print("Loading SVHN dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = torchvision.datasets.SVHN(
        root='./data', split='train', download=True, transform=transform)
    test_dataset = torchvision.datasets.SVHN(
        root='./data', split='test', download=True, transform=transform)
    
    return train_dataset, test_dataset

def generate_synthetic_data(num_samples=1000, num_classes=5, num_dim=60, num_clusters=2):
    """Generate synthetic data with clusters."""
    print(f"Generating synthetic data with {num_samples} samples, {num_classes} classes, {num_dim} dimensions, {num_clusters} clusters...")
    
    from torch.utils.data import TensorDataset
    
    # Generate means for each class
    means = np.random.randn(num_classes, num_dim)
    
    # Generate cluster means for each class
    cluster_means = np.random.randn(num_classes, num_clusters, num_dim) * 0.5
    
    # Generate covariance matrices for each class
    covs = []
    for _ in range(num_classes):
        scale_matrix = np.eye(num_dim)
        cov = wishart.rvs(df=num_dim, scale=scale_matrix, size=1)
        # Make it positive semi-definite
        cov = cov @ cov.T
        # Scale it down
        cov = cov / 100
        covs.append(cov)
    
    # Generate data for each class
    data = []
    labels = []
    samples_per_class = num_samples // num_classes
    
    for c in range(num_classes):
        for _ in range(samples_per_class):
            # Randomly choose a cluster
            cluster_idx = np.random.randint(0, num_clusters)
            # Sample from the Gaussian distribution of this class and cluster
            class_mean = means[c] + cluster_means[c, cluster_idx]
            sample = np.random.multivariate_normal(class_mean, covs[c])
            data.append(sample)
            labels.append(c)
    
    # Convert to PyTorch tensors
    data = torch.tensor(np.array(data), dtype=torch.float32)
    labels = torch.tensor(np.array(labels), dtype=torch.long)
    
    # Create TensorDataset
    dataset = TensorDataset(data, labels)
    
    # Split into train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    return train_dataset, test_dataset

def load_celeba(data_dir='./data/celeba'):
    """Load preprocessed CelebA dataset."""
    print("Loading CelebA dataset...")
    
    # Define transforms for CelebA
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Check if dataset exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"CelebA dataset not found at {data_dir}. Please preprocess the dataset first.")
    
    # Use a custom dataset for CelebA with preprocessed images
    class CelebADataset(torch.utils.data.Dataset):
        def __init__(self, data_dir, split='train', transform=None):
            self.data_dir = data_dir
            self.transform = transform
            self.split = split
            
            # Load preprocessed data
            data_path = os.path.join(data_dir, f"celeba_{split}.pt")
            self.data, self.labels = torch.load(data_path)
            
        def __len__(self):
            return len(self.data)
            
        def __getitem__(self, idx):
            img = self.data[idx].float() / 255.0
            label = self.labels[idx]
            
            if self.transform:
                img = self.transform(img)
                
            return img, label
    
    # Create train and test datasets
    train_dataset = CelebADataset(data_dir, split='train', transform=transform)
    test_dataset = CelebADataset(data_dir, split='test', transform=transform)
    
    return train_dataset, test_dataset

def load_dataset(dataset_name):
    """Load dataset based on name."""
    if dataset_name == 'mnist':
        return load_mnist()
    elif dataset_name == 'cifar10':
        return load_cifar10()
    elif dataset_name == 'cifar100':
        return load_cifar100()
    elif dataset_name == 'svhn':
        return load_svhn()
    elif dataset_name == 'synthetic':
        return generate_synthetic_data()
    elif dataset_name == 'celeba':
        return load_celeba()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}") 