import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from scipy.stats import wishart
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import urllib.request
import zipfile
import io
import re
import string
import json
import csv
from collections import Counter
import sys

# Define a flag to skip torchtext import entirely - set to True
SKIP_TORCHTEXT = True

# Only try importing torchtext if not skipped
if not SKIP_TORCHTEXT:
    try:
        import torchtext
        from torchtext.data.utils import get_tokenizer
        from torchtext.vocab import build_vocab_from_iterator
        from torch.nn.utils.rnn import pad_sequence
        torchtext_available = True
        print("Successfully imported torchtext")
    except ImportError:
        torchtext_available = False
        print("Warning: torchtext not found. AG News and IMDb datasets will not be available.")
        print("Install with: pip install torchtext")
    except OSError as e:
        # This catches the "undefined symbol" error that happens when torchtext is incompatible with torch
        torchtext_available = False
        print(f"Warning: torchtext found but incompatible with current torch version: {e}")
        print("Using direct implementations for AG News and IMDb")
else:
    # Skip torchtext entirely
    torchtext_available = False
    print("Skipping torchtext import - using direct dataset implementations")

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

def load_fashion_mnist():
    """Load Fashion-MNIST dataset."""
    print("Loading Fashion-MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2861,), (0.3530,))  # Fashion-MNIST specific normalization
    ])
    
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST(
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
    """Load CIFAR-100 dataset with AutoAugment for improved training."""
    print("Loading CIFAR-100 dataset with AutoAugment...")
    
    try:
        from torchvision.transforms import autoaugment
        
        # Enhanced training transform with AutoAugment
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            autoaugment.AutoAugment(autoaugment.AutoAugmentPolicy.CIFAR10),  # Use CIFAR10 policy for CIFAR100
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        print("AutoAugment successfully applied to CIFAR-100")
    except (ImportError, AttributeError) as e:
        print(f"Warning: AutoAugment not available ({str(e)}). Using standard augmentation for CIFAR-100.")
        # Fallback to standard augmentation if AutoAugment is not available
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    
    # Test transform remains the same
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

class ShakespeareDataset(Dataset):
    def __init__(self, data_dir, split='train', seq_length=80):
        self.seq_length = seq_length
        
        # Read the text file
        file_path = os.path.join(data_dir, f'shakespeare_{split}.txt')
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # Create character to index mapping
        chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        
        # Convert text to indices
        self.data = torch.tensor([self.char_to_idx[ch] for ch in text])
        
        # Create sequences
        self.sequences = []
        self.targets = []
        for i in range(0, len(self.data) - self.seq_length):
            self.sequences.append(self.data[i:i + self.seq_length])
            self.targets.append(self.data[i + 1:i + self.seq_length + 1])
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def load_shakespeare(data_dir='./data/shakespeare'):
    """Load Shakespeare dataset."""
    print("Loading Shakespeare dataset...")
    
    # Check if dataset exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Shakespeare dataset not found at {data_dir}. Please prepare the dataset first.")
    
    # Create train and test datasets
    train_dataset = ShakespeareDataset(data_dir, split='train')
    test_dataset = ShakespeareDataset(data_dir, split='test')
    
    return train_dataset, test_dataset

def load_adult():
    """Load Adult Census Income dataset."""
    print("Loading Adult Census Income dataset...")
    
    # Create data directory if it doesn't exist
    os.makedirs('./data/adult', exist_ok=True)
    
    # Define URLs
    train_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    test_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
    
    # Define column names
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]
    
    # Define categorical and numerical columns
    categorical_columns = [
        'workclass', 'education', 'marital-status', 'occupation',
        'relationship', 'race', 'sex', 'native-country'
    ]
    numerical_columns = [
        'age', 'fnlwgt', 'education-num', 'capital-gain',
        'capital-loss', 'hours-per-week'
    ]
    
    try:
        # Download and load the dataset
        train_data = pd.read_csv(train_url, names=columns, skipinitialspace=True)
        test_data = pd.read_csv(test_url, names=columns, skipinitialspace=True, skiprows=1)
        
        # Remove the '?' values
        train_data = train_data.replace('?', np.nan)
        test_data = test_data.replace('?', np.nan)
        
        # Drop rows with missing values
        train_data = train_data.dropna()
        test_data = test_data.dropna()
        
        # Convert income to binary (0 for <=50K, 1 for >50K)
        train_data['income'] = (train_data['income'].str.contains('>50K')).astype(int)
        test_data['income'] = (test_data['income'].str.contains('>50K')).astype(int)
        
        # Create label encoders for categorical variables
        label_encoders = {}
        for column in categorical_columns:
            label_encoders[column] = LabelEncoder()
            train_data[column] = label_encoders[column].fit_transform(train_data[column])
            test_data[column] = label_encoders[column].transform(test_data[column])
        
        # Normalize numerical features
        scaler = StandardScaler()
        train_data[numerical_columns] = scaler.fit_transform(train_data[numerical_columns])
        test_data[numerical_columns] = scaler.transform(test_data[numerical_columns])
        
        # Create custom dataset class
        class AdultDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.features = torch.FloatTensor(data[numerical_columns + categorical_columns].values)
                self.labels = torch.LongTensor(data['income'].values)
                
            def __len__(self):
                return len(self.features)
                
            def __getitem__(self, idx):
                return self.features[idx], self.labels[idx]
        
        # Create train and test datasets
        train_dataset = AdultDataset(train_data)
        test_dataset = AdultDataset(test_data)
        
        print(f"Adult dataset loaded successfully. Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
        return train_dataset, test_dataset
        
    except Exception as e:
        print(f"Error loading Adult dataset: {str(e)}")
        raise

def load_covertype():
    """Load Forest Covertype dataset."""
    print("Loading Forest Covertype dataset...")
    
    # Create data directory if it doesn't exist
    os.makedirs('./data/covertype', exist_ok=True)
    
    # Define URL
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz'
    local_path = './data/covertype/covertype.data.gz'
    
    try:
        # Define column names: 54 features + class label
        feature_names = [f'feature_{i}' for i in range(1, 55)]
        columns = feature_names + ['class']
        
        # Check if the data file exists locally
        if not os.path.exists(local_path):
            print(f"Downloading Covertype dataset from {url}...")
            import urllib.request
            import gzip
            import shutil
            
            # Download the file
            urllib.request.urlretrieve(url, local_path)
            
            # Extract the gz file
            with gzip.open(local_path, 'rb') as f_in:
                with open('./data/covertype/covertype.data', 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            print("Download and extraction completed.")
        
        # Load the data
        data = pd.read_csv('./data/covertype/covertype.data', header=None, names=columns)
        
        # Class labels are 1-7, convert to 0-6 for PyTorch
        data['class'] = data['class'] - 1
        
        # Split into features and labels
        X = data.drop('class', axis=1)
        y = data['class']
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split into train and test
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create custom dataset class
        class CovertypeDataset(torch.utils.data.Dataset):
            def __init__(self, features, labels):
                self.features = torch.FloatTensor(features)
                self.labels = torch.LongTensor(labels.values)
                
            def __len__(self):
                return len(self.features)
                
            def __getitem__(self, idx):
                return self.features[idx], self.labels[idx]
        
        # Create train and test datasets
        train_dataset = CovertypeDataset(X_train, y_train)
        test_dataset = CovertypeDataset(X_test, y_test)
        
        print(f"Covertype dataset loaded successfully. Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
        return train_dataset, test_dataset
        
    except Exception as e:
        print(f"Error loading Covertype dataset: {str(e)}")
        raise

def load_credit():
    """Load Credit Card Fraud Detection dataset."""
    print("Loading Credit Card Fraud Detection dataset...")
    
    # Create data directory if it doesn't exist
    os.makedirs('./data/creditcard', exist_ok=True)
    
    # Define URL
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv'
    local_path = './data/creditcard/creditcard.csv'
    
    try:
        # Check if the data file exists locally
        if not os.path.exists(local_path):
            print(f"Downloading Credit Card Fraud dataset from {url}...")
            import urllib.request
            
            # Download the file
            urllib.request.urlretrieve(url, local_path)
            print("Download completed.")
        
        # Load the data
        data = pd.read_csv(local_path)
        
        # Split into features and labels
        X = data.drop(['Class', 'Time'], axis=1)  # Drop Time as it's not relevant for prediction
        y = data['Class']
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Handle class imbalance with stratified sampling
        # Since fraud cases (class 1) are very rare (0.17%)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create custom dataset class
        class CreditCardDataset(torch.utils.data.Dataset):
            def __init__(self, features, labels):
                self.features = torch.FloatTensor(features)
                # Ensure labels are integer type before converting to LongTensor
                self.labels = torch.tensor(labels.values.astype(np.int64), dtype=torch.long)
                
            def __len__(self):
                return len(self.features)
                
            def __getitem__(self, idx):
                return self.features[idx], self.labels[idx]
        
        # Create train and test datasets
        train_dataset = CreditCardDataset(X_train, y_train)
        test_dataset = CreditCardDataset(X_test, y_test)
        
        print(f"Credit Card dataset loaded successfully. Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
        return train_dataset, test_dataset
        
    except Exception as e:
        print(f"Error loading Credit Card dataset: {str(e)}")
        raise

def load_kdd_cup_99():
    """Load KDD Cup 99 network intrusion detection dataset."""
    print("Loading KDD Cup 99 dataset...")
    
    # Create data directory if it doesn't exist
    os.makedirs('./data/kddcup99', exist_ok=True)
    
    # Use the correct URL from the KDD site
    url = 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz'
    local_path_gz = './data/kddcup99/kddcup.data_10_percent.gz'
    local_path = './data/kddcup99/kddcup.data_10_percent'
    
    try:
        # Check if the data file exists locally
        if not os.path.exists(local_path):
            print(f"Downloading KDD Cup 99 dataset from {url}...")
            import urllib.request
            import gzip
            
            try:
                # Download the file
                urllib.request.urlretrieve(url, local_path_gz)
                
                # Extract the gz file
                print("Extracting dataset...")
                with gzip.open(local_path_gz, 'rb') as f_in:
                    with open(local_path, 'wb') as f_out:
                        f_out.write(f_in.read())
                
                # Remove the gz file to save space
                os.remove(local_path_gz)
            except Exception as download_error:
                print(f"Download from {url} failed: {download_error}")
                print("Attempting alternative method using ucimlrepo...")
                
                # Try using ucimlrepo as fallback
                try:
                    import subprocess
                    subprocess.run([sys.executable, "-m", "pip", "install", "ucimlrepo"], check=True)
                    
                    from ucimlrepo import fetch_ucirepo
                    kdd_data = fetch_ucirepo(id=130)
                    
                    # Convert to pandas and save
                    X = kdd_data.data.features
                    y = kdd_data.data.targets
                    
                    # Save as CSV for consistency
                    df = pd.concat([X, y], axis=1)
                    df.to_csv(local_path, index=False, header=False)
                    print("Downloaded using ucimlrepo package")
                except Exception as uci_error:
                    raise Exception(f"Both download methods failed. Original error: {download_error}, UCI error: {uci_error}")
        
        print("Processing KDD Cup 99 dataset...")

        # Define column names
        columns = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
        ]
        
        # Load the data
        data = pd.read_csv(local_path, header=None, names=columns)
        
        # Map labels to 5 main categories (0: normal, 1-4: attack types)
        attack_mapping = {
            'normal.': 0,
            # DOS attacks
            'back.': 1, 'land.': 1, 'neptune.': 1, 'pod.': 1, 'smurf.': 1, 'teardrop.': 1,
            # R2L attacks
            'ftp_write.': 2, 'guess_passwd.': 2, 'imap.': 2, 'multihop.': 2, 
            'phf.': 2, 'spy.': 2, 'warezclient.': 2, 'warezmaster.': 2,
            # U2R attacks
            'buffer_overflow.': 3, 'loadmodule.': 3, 'perl.': 3, 'rootkit.': 3,
            # Probe attacks
            'ipsweep.': 4, 'nmap.': 4, 'portsweep.': 4, 'satan.': 4
        }
        
        # Map unknown attacks to category 1 (DOS) as default
        data['label'] = data['label'].map(lambda x: attack_mapping.get(x, 1))
        
        # Identify categorical and numerical columns
        categorical_columns = ['protocol_type', 'service', 'flag']
        numerical_columns = [col for col in columns[:-1] if col not in categorical_columns]
        
        # Encode categorical variables
        label_encoders = {}
        for column in categorical_columns:
            label_encoders[column] = LabelEncoder()
            data[column] = label_encoders[column].fit_transform(data[column])
        
        # Normalize numerical features
        scaler = StandardScaler()
        data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
        
        # Split features and labels
        X = data.drop('label', axis=1)
        y = data['label']
        
        # Split into train and test with stratification
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create custom dataset class
        class KDDDataset(torch.utils.data.Dataset):
            def __init__(self, features, labels):
                self.features = torch.FloatTensor(features.values)
                self.labels = torch.LongTensor(labels.values)
                
            def __len__(self):
                return len(self.features)
                
            def __getitem__(self, idx):
                return self.features[idx], self.labels[idx]
        
        # Create train and test datasets
        train_dataset = KDDDataset(X_train, y_train)
        test_dataset = KDDDataset(X_test, y_test)
        
        print(f"KDD Cup 99 dataset loaded successfully. Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
        print(f"Classes: 0=Normal, 1=DOS, 2=R2L, 3=U2R, 4=Probe")
        return train_dataset, test_dataset
        
    except Exception as e:
        print(f"Error loading KDD Cup 99 dataset: {str(e)}")
        raise

# --- Custom Dataset Classes for Text ---

class TextClassificationDataset(Dataset):
    """
    Generic Dataset for text classification tasks using torchtext iterators.
    Handles tokenization, numericalization, and vocabulary building.
    """
    def __init__(self, dataset_iterator, tokenizer, vocab):
        self.tokenizer = tokenizer
        self.vocab = vocab
        self._data = []
        self._labels = []

        print(f"Processing text data...")
        for label, text in dataset_iterator:
            # Labels in AG_NEWS are 1-4, need 0-3
            # Labels in IMDb seem to be 1-2? Need 0-1
            # Adjust labels to be 0-indexed
            processed_label = int(label) - 1

            tokens = self.tokenizer(text)
            token_indices = self.vocab(tokens) # Convert tokens to indices

            # Store as list of ints, convert to tensor later in collate_fn
            self._data.append(torch.tensor(token_indices, dtype=torch.long))
            self._labels.append(torch.tensor(processed_label, dtype=torch.long))
        print(f"Finished processing {len(self._data)} items.")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx], self._labels[idx]

    def get_vocab(self):
        return self.vocab

    def get_vocab_size(self):
        return len(self.vocab)

# --- AG News Loader ---

def load_ag_news(data_dir='./data'):
    """Load AG News dataset using torchtext."""
    if not torchtext_available:
        raise ImportError("torchtext is required to load the AG News dataset. Since torchtext is disabled or unavailable, please use the direct implementation by calling load_ag_news_direct() instead.")

    print("Loading AG News dataset...")
    tokenizer = get_tokenizer('basic_english')
    
    # Load raw data iterators
    try:
        train_iter, test_iter = torchtext.datasets.AG_NEWS(root=data_dir, split=('train', 'test'))
    except Exception as e:
        print(f"\nError loading AG_NEWS. It might be downloading.")
        print("If download fails, check network connection or try manually downloading from:")
        print("https://pytorch.org/text/stable/datasets.html#ag-news")
        print(f"Specific error: {e}")
        raise

    # Build vocabulary from training data
    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)

    print("Building AG News vocabulary...")
    # Need to reset train_iter after using it for vocab building
    train_iter_for_vocab, _ = torchtext.datasets.AG_NEWS(root=data_dir, split=('train', 'test'))
    vocab = build_vocab_from_iterator(yield_tokens(train_iter_for_vocab), specials=["<unk>", "<pad>"])
    vocab.set_default_index(vocab["<unk>"])
    print(f"Vocabulary built. Size: {len(vocab)}")

    # Create Dataset objects (handle tokenization/numericalization inside)
    # Reset iterators again before passing to Dataset class
    train_iter, test_iter = torchtext.datasets.AG_NEWS(root=data_dir, split=('train', 'test'))
    train_dataset = TextClassificationDataset(train_iter, tokenizer, vocab)
    test_dataset = TextClassificationDataset(test_iter, tokenizer, vocab) # Use same vocab

    # Add vocab size attribute for model creation convenience
    train_dataset.vocab_size = len(vocab)
    test_dataset.vocab_size = len(vocab)
    train_dataset.num_classes = 4 # AG News has 4 classes
    test_dataset.num_classes = 4

    return train_dataset, test_dataset

# --- IMDb Loader ---

# def load_imdb(data_dir='./data'):
#     """Load IMDb dataset using torchtext."""
#     if not torchtext_available:
#         raise ImportError("torchtext is required to load the IMDb dataset. Since torchtext is disabled or unavailable, please use the direct implementation by calling load_imdb_direct() instead.")
#     
#     # Rest of the function remains unchanged...
#     print("Loading IMDb dataset...")
# ... (rest of load_imdb function removed)

def load_dataset(dataset_name):
    """Load the specified dataset."""
    if dataset_name == 'mnist':
        return load_mnist()
    elif dataset_name == 'fashion_mnist':
        return load_fashion_mnist()
    elif dataset_name == 'cifar10':
        return load_cifar10()
    elif dataset_name == 'cifar100':
        return load_cifar100()
    elif dataset_name == 'svhn':
        return load_svhn()
    elif dataset_name == 'celeba':
        return load_celeba()
    elif dataset_name == 'shakespeare':
        return load_shakespeare()
    elif dataset_name == 'adult':
        return load_adult()
    elif dataset_name == 'covertype':
        return load_covertype()
    elif dataset_name == 'credit':
        return load_credit()
    elif dataset_name == 'kdd_cup_99':
        return load_kdd_cup_99()
    elif dataset_name == 'ag_news':
        # Always use direct implementation to avoid torchtext issues
        return load_ag_news_direct()
    elif dataset_name == 'newsgroups':
        return load_newsgroups()
    # REMOVED IMDB CASE
    # elif dataset_name == 'imdb':
    #     # Always use direct implementation to avoid torchtext issues
    #     return load_imdb_direct()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

# --- Collate Function for Text Data (using EmbeddingBag) ---
def collate_batch(batch):
    """
    Collate batch for text data - convert to format suitable for EmbeddingBag.
    Now handles both tuple format (data, label) and single tensor format.
    """
    text_list, label_list = [], []
    lengths = []
    num_items = len(batch)
    
    if not batch:
        # Return empty default tensors with the right shapes
        return (torch.tensor([], dtype=torch.long), 
                torch.tensor([], dtype=torch.long), 
                torch.tensor([0], dtype=torch.long))  # Offsets must contain at least one element

    for item in batch:
        if isinstance(item, tuple) and len(item) == 2:
            # This is the normal (data, label) format
            data_tensor, label_tensor = item
            
            if isinstance(data_tensor, torch.Tensor) and isinstance(label_tensor, torch.Tensor):
                text_list.append(data_tensor)
                label_list.append(label_tensor)
                lengths.append(data_tensor.size(0))
        elif isinstance(item, torch.Tensor):
            # This is a single tensor format (assume it's data)
            text_list.append(item)
            # Create a dummy label of 0
            label_list.append(torch.tensor(0, dtype=torch.long))
            lengths.append(item.size(0))

    # Ensure we have at least one valid item
    if not text_list:
        # Return empty default tensors with the right shapes
        return (torch.tensor([], dtype=torch.long), 
                torch.tensor([], dtype=torch.long), 
                torch.tensor([0], dtype=torch.long))

    # Ensure we have at least one element in each text tensor
    valid_text_list = []
    valid_label_list = []
    valid_lengths = []
    for i, (text, label, length) in enumerate(zip(text_list, label_list, lengths)):
        if length > 0:
            valid_text_list.append(text)
            valid_label_list.append(label)
            valid_lengths.append(length)
    
    # If all tensors were empty, return default tensors
    if not valid_text_list:
        return (torch.tensor([], dtype=torch.long), 
                torch.tensor([], dtype=torch.long), 
                torch.tensor([0], dtype=torch.long))
    
    # Concatenate all text tensors and create offsets
    try:
        # Stack labels as a 1D tensor
        labels = torch.stack(valid_label_list)
        
        # Concatenate text tensors
        text = torch.cat(valid_text_list)
        
        # Create offsets for EmbeddingBag
        offsets = torch.zeros(len(valid_lengths), dtype=torch.long)
        if len(valid_lengths) > 1:
            offsets[1:] = torch.tensor(valid_lengths[:-1]).cumsum(0)
    
        return labels, text, offsets
    except Exception:
        # Fall back to default tensors
        return (torch.tensor([], dtype=torch.long), 
                torch.tensor([], dtype=torch.long), 
                torch.tensor([0], dtype=torch.long))

# --- Direct Implementation for AG News Dataset ---
class CustomVocab:
    """Simple vocabulary class to replace torchtext.vocab.Vocab."""
    def __init__(self, word_freqs, min_freq=1, specials=["<unk>", "<pad>"]):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_freqs = word_freqs
        
        # Add special tokens first
        idx = 0
        for special in specials:
            self.word_to_idx[special] = idx
            self.idx_to_word[idx] = special
            idx += 1
            
        # Add regular words that meet min_freq
        for word, freq in word_freqs.items():
            if freq >= min_freq and word not in self.word_to_idx:
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
                idx += 1
        
        # Default unknown index
        self.unk_idx = self.word_to_idx.get("<unk>", 0)
        
    def __getitem__(self, word):
        """Get index of a word, or return unk_idx if not found."""
        if isinstance(word, list):
            # For batch conversion
            return [self[w] for w in word]
        return self.word_to_idx.get(word, self.unk_idx)
    
    def __len__(self):
        return len(self.word_to_idx)

class CustomTextDataset(Dataset):
    """Simple text dataset class to replace TextClassificationDataset."""
    def __init__(self, texts, labels, tokenizer, vocab, max_len=None):
        # Store raw texts and labels
        self.texts = texts
        # Store the original labels (might be list or tensor)
        self.original_labels = labels 
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_len = max_len
        self.vocab_size = len(vocab)
        self.num_classes = len(set([l.item() for l in labels])) if isinstance(labels[0], torch.Tensor) else len(set(labels))
        
        # --- Initialize _data attribute --- 
        # Although we process in __getitem__, evaluation logic checks for its existence.
        self._data = [] 
        # --- End Initialize _data --- 
        
        # --- Re-add self._labels --- 
        # Store processed label tensors for compatibility with main.py's client creation
        self._labels = []
        for label in labels:
            label_tensor = label if isinstance(label, torch.Tensor) else torch.tensor(label, dtype=torch.long)
            self._labels.append(label_tensor)
        # --- End Re-add --- 

        print(f"CustomTextDataset initialized for {len(self.texts)} items. Processed {len(self._labels)} labels.")
    
    def __len__(self):
        # Length is based on the original texts list
        return len(self.texts)
    
    def __getitem__(self, idx):
        # Process the item when it's requested by the DataLoader
        text = self.texts[idx]
        label = self.original_labels[idx] 
        
        tokens = self.tokenizer(text)
        token_indices = self.vocab[tokens] 
        
        # Apply index clamping
        valid_indices = []
        max_valid_index = self.vocab_size - 1
        unk_idx = self.vocab.unk_idx
        
        for index in token_indices:
            if index > max_valid_index:
                print(f"[GetItem Warning] Index {index} >= vocab_size {self.vocab_size}. Mapping to UNK ({unk_idx}). Text: '{text[:50]}...'")
                valid_indices.append(unk_idx) 
            elif index < 0:
                print(f"[GetItem Warning] Negative index {index} found. Mapping to UNK ({unk_idx}). Text: '{text[:50]}...'")
                valid_indices.append(unk_idx)
            else:
                valid_indices.append(index)
        
        # Optional padding/truncation
        if self.max_len:
            pad_idx = self.vocab["<pad>"]
            if len(valid_indices) < self.max_len:
                valid_indices += [pad_idx] * (self.max_len - len(valid_indices))
            else:
                valid_indices = valid_indices[:self.max_len]
        
        # Convert to tensor
        data_tensor = torch.tensor(valid_indices, dtype=torch.long)
        label_tensor = label if isinstance(label, torch.Tensor) else torch.tensor(label, dtype=torch.long)

        # Return the tuple expected by collate_batch
        return data_tensor, label_tensor

def basic_tokenizer(text):
    """Simple tokenizer function to replace torchtext's get_tokenizer."""
    # Convert to lowercase
    text = text.lower()
    # Replace punctuation with spaces
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    # Split on whitespace and filter empty strings
    tokens = [token for token in text.split() if token]
    return tokens

def download_file(url, filename=None):
    """Download a file from a URL and return its contents."""
    print(f"Downloading from {url}...")
    
    # Create data directory if it doesn't exist
    os.makedirs('./data', exist_ok=True)
    
    if filename:
        # Download to a file if filename is provided
        filepath = os.path.join('./data', filename)
        if not os.path.exists(filepath):
            urllib.request.urlretrieve(url, filepath)
            print(f"Downloaded to {filepath}")
        else:
            print(f"File already exists: {filepath}")
        return filepath
    else:
        # Return content directly if no filename
        with urllib.request.urlopen(url) as response:
            return response.read()

def load_ag_news_direct():
    """Load AG News dataset without using torchtext."""
    print("Loading AG News dataset directly (without torchtext)...")
    
    # Create AG News directory
    data_dir = os.path.join('./data', 'ag_news')
    os.makedirs(data_dir, exist_ok=True)
    
    # URLs for AG News dataset
    train_url = 'https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv'
    test_url = 'https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv'
    
    # Download files
    train_file = download_file(train_url, 'ag_news_train.csv')
    test_file = download_file(test_url, 'ag_news_test.csv')
    
    print(f"Train file path: {os.path.abspath(train_file)}")
    print(f"Test file path: {os.path.abspath(test_file)}")
    print(f"Checking if files exist - Train: {os.path.exists(train_file)}, Test: {os.path.exists(test_file)}")
    
    # Read CSV files
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []
    
    print("Reading train data...")
    with open(train_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, quotechar='"')
        for row in reader:
            # Format: class index (1-4), title, description
            label = int(row[0]) - 1  # Convert to 0-3
            # Combine title and description
            text = row[1] + " " + row[2]
            train_texts.append(text)
            train_labels.append(label)
    
    print("Reading test data...")
    with open(test_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, quotechar='"')
        for row in reader:
            label = int(row[0]) - 1  # Convert to 0-3
            text = row[1] + " " + row[2]
            test_texts.append(text)
            test_labels.append(label)
    
    print(f"Read {len(train_texts)} training samples and {len(test_texts)} test samples")
    
    # Create tokenizer
    tokenizer = basic_tokenizer
    
    # Build vocabulary from training data
    word_freqs = Counter()
    print("Building vocabulary from training data...")
    for text in train_texts:
        tokens = tokenizer(text)
        word_freqs.update(tokens)
    
    # Create vocabulary
    vocab = CustomVocab(word_freqs, min_freq=5, specials=["<unk>", "<pad>"])
    print(f"Built vocabulary with {len(vocab)} tokens")
    
    # Create tensors for labels
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_labels = torch.tensor(test_labels, dtype=torch.long)
    
    # Create datasets
    train_dataset = CustomTextDataset(train_texts, train_labels, tokenizer, vocab)
    test_dataset = CustomTextDataset(test_texts, test_labels, tokenizer, vocab)
    
    print(f"AG News datasets created. Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    print(f"Number of classes: {train_dataset.num_classes}, Vocabulary size: {train_dataset.vocab_size}")
    return train_dataset, test_dataset

def load_newsgroups():
    """Load 20 Newsgroups dataset (full set, not subset)."""
    print("Loading 20 Newsgroups dataset (full set)...")
    
    # Create newsgroups directory
    data_dir = os.path.join('./data', 'newsgroups')
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        from sklearn.datasets import fetch_20newsgroups
        
        # Load train and test data
        print("Fetching training data...")
        newsgroups_train = fetch_20newsgroups(
            subset='train',
            remove=('headers', 'footers', 'quotes'),
            random_state=42,
            data_home=data_dir
        )
        
        print("Fetching test data...")
        newsgroups_test = fetch_20newsgroups(
            subset='test',
            remove=('headers', 'footers', 'quotes'),
            random_state=42,
            data_home=data_dir
        )
        
        # Extract texts and labels
        train_texts = newsgroups_train.data
        train_labels = newsgroups_train.target
        test_texts = newsgroups_test.data
        test_labels = newsgroups_test.target
        
        print(f"Loaded {len(train_texts)} training samples and {len(test_texts)} test samples")
        print(f"Categories: {newsgroups_train.target_names}")
        
        # Create tokenizer
        tokenizer = basic_tokenizer
        
        # Build vocabulary from training data
        word_freqs = Counter()
        print("Building vocabulary from training data...")
        for text in train_texts:
            tokens = tokenizer(text)
            word_freqs.update(tokens)
        
        # Create vocabulary with higher min_freq for this larger dataset
        vocab = CustomVocab(word_freqs, min_freq=10, specials=["<unk>", "<pad>"])
        print(f"Built vocabulary with {len(vocab)} tokens")
        
        # Convert labels to tensors
        train_labels = torch.tensor(train_labels, dtype=torch.long)
        test_labels = torch.tensor(test_labels, dtype=torch.long)
        
        # Create datasets
        train_dataset = CustomTextDataset(train_texts, train_labels, tokenizer, vocab)
        test_dataset = CustomTextDataset(test_texts, test_labels, tokenizer, vocab)
        
        # Set num_classes to 20 for 20 newsgroups
        train_dataset.num_classes = 20
        test_dataset.num_classes = 20
        
        print(f"20 Newsgroups datasets created. Train: {len(train_dataset)}, Test: {len(test_dataset)}")
        print(f"Number of classes: {train_dataset.num_classes}, Vocabulary size: {train_dataset.vocab_size}")
        
        return train_dataset, test_dataset
        
    except ImportError:
        print("scikit-learn is required for 20 Newsgroups dataset. Please install it with: pip install scikit-learn")
        raise

