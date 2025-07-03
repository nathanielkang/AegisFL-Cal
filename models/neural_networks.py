import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models # For ResNet

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10)
        self.num_classes = 10
        self.fc = self.fc2

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class CIFAR10Net(nn.Module):
    def __init__(self):
        super(CIFAR10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)
        self.num_classes = 10
        self.fc = self.fc2

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CIFAR100Net(nn.Module):
    def __init__(self):
        super(CIFAR100Net, self).__init__()
        
        # Flag to easily identify this as a CIFAR-100 model
        self.is_cifar100 = True
        self.num_classes = 100
        
        try:
            import torchvision.models as models
            
            # Load a pretrained ResNet18 with ImageNet weights
            print("Loading pretrained ResNet18 from torchvision...")
            self.backbone = models.resnet18(weights='IMAGENET1K_V1')
            print("Pretrained model loaded successfully")
            
            # Modify the first conv layer for CIFAR-100 (32x32 images)
            self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            
            # Replace maxpool with a more suitable pooling for CIFAR-100
            self.backbone.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            
            # Add batch normalization after the first conv layer
            self.bn1 = nn.BatchNorm2d(64)
            
            # Change the final classification layer to output 100 classes
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 100)
            )
            
            # Freeze early layers to prevent overfitting
            for param in list(self.backbone.parameters())[:-6]:  # Keep last 2 layers trainable
                param.requires_grad = False
                
            print(f"CIFAR100Net initialized with pretrained weights - {self._count_parameters():,} parameters")
            print(f"Final classification layer: {num_features} -> 512 -> 100")
            
        except Exception as e:
            print(f"Error loading pretrained model: {e}")
            print("Falling back to a more robust CNN architecture")
            
            # Improved fallback architecture
            self.backbone = nn.Sequential(
                # First conv block
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.25),
                
                # Second conv block
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.25),
                
                # Third conv block
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.25),
                
                # Fourth conv block
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.25),
                
                # Global average pooling and classifier
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 100)
            )
    
    def _count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x):
        if hasattr(self, 'bn1'):
            x = self.backbone.conv1(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.backbone.fc(x)
        else:
            x = self.backbone(x)
        return x

class SimplifiedBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SimplifiedBlock, self).__init__()
        
        # Simplified residual block - just 2 conv layers with BN and ReLU
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Handle downsample if needed
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add identity connection and apply ReLU
        out += identity
        out = self.relu(out)
        
        return out

class SVHNNet(nn.Module):
    def __init__(self):
        super(SVHNNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)
        self.num_classes = 10
        self.fc = self.fc2

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CelebANet(nn.Module):
    def __init__(self):
        super(CelebANet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 2)  # Binary classification
        self.num_classes = 2
        self.fc = self.fc2

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(-1, 256 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ShakespeareLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.5):
        super(ShakespeareLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.num_classes = vocab_size
        
    def forward(self, x):
        # x shape: (batch_size, seq_length)
        batch_size = x.size(0)
        
        # Embedding layer
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # LSTM layer
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_length, hidden_dim)
        
        # Reshape output for fully connected layer
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        # Fully connected layer
        out = self.fc(lstm_out)  # (batch_size * seq_length, vocab_size)
        
        return out  # Remove log_softmax as we're using CrossEntropyLoss 

class AdultNet(nn.Module):
    def __init__(self, input_dim=14):
        super(AdultNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # Binary classification
        self.num_classes = 2
        self.fc = self.fc3
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc3(x)
        return x

class CovertypeNet(nn.Module):
    def __init__(self, input_dim=54):
        super(CovertypeNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 7)  # 7 forest cover types
        self.dropout = nn.Dropout(0.3)
        
        # Add num_classes attribute to make compatible with FedAlignLoss
        self.num_classes = 7
        # Also add a reference to fc to match common model pattern
        self.fc = self.fc4
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

class CreditNet(nn.Module):
    def __init__(self, input_dim=29):  # 30 features minus Time
        super(CreditNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 2)  # Binary classification (fraud or not)
        self.dropout = nn.Dropout(0.2)
        self.num_classes = 2
        self.fc = self.fc4
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

class KDDNet(nn.Module):
    def __init__(self, input_dim=41):  # 41 features for KDD Cup 99
        super(KDDNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 5)  # 5 classes (normal + 4 attack types)
        self.dropout = nn.Dropout(0.3)
        self.num_classes = 5
        self.fc = self.fc4
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

# Placeholder VAE for FedFed
class GenericVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(GenericVAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc21 = nn.Linear(400, latent_dim)  # mu layer
        self.fc22 = nn.Linear(400, latent_dim)  # logvar layer
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))  # Use sigmoid for reconstruction if input is normalized [0,1]

    def forward(self, x):
        # Flatten input if needed (e.g., for image data passed as flat vector)
        if x.dim() > 2:
             x = x.view(x.size(0), -1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Model for AG News
class AGNewsNet(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_class=4):
        super(AGNewsNet, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False) # sparse=True might be faster but check compatibility
        self.fc = nn.Linear(embed_dim, num_class)
        self._init_weights()
        self.num_classes = num_class

    def _init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        # Check if input is empty to avoid runtime errors
        if text.numel() == 0:
            # Return a default tensor with correct shape (batch_size, num_class)
            batch_size = offsets.size(0) if offsets.numel() > 0 else 0
            if batch_size > 0:
                return torch.zeros((batch_size, self.fc.out_features), device=text.device)
            else:
                # If we can't determine batch size, return empty tensor with correct shape
                return torch.zeros((0, self.fc.out_features), device=text.device)
                
        # Regular forward pass
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

# Model for 20 Newsgroups
class NewsGroupsNet(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_class=20):
        super(NewsGroupsNet, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc1 = nn.Linear(embed_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_class)
        self.dropout = nn.Dropout(0.5)
        self._init_weights()
        self.num_classes = num_class
        self.fc = self.fc3

    def _init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()
        self.fc3.weight.data.uniform_(-initrange, initrange)
        self.fc3.bias.data.zero_()

    def forward(self, text, offsets):
        # Check if input is empty to avoid runtime errors
        if text.numel() == 0:
            batch_size = offsets.size(0) if offsets.numel() > 0 else 0
            if batch_size > 0:
                return torch.zeros((batch_size, self.fc3.out_features), device=text.device)
            else:
                return torch.zeros((0, self.fc3.out_features), device=text.device)
                
        # Regular forward pass
        embedded = self.embedding(text, offsets)
        x = F.relu(self.fc1(embedded))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

# --- Lightweight Feature Extractor for PCA on Images ---
class SimpleImagePCALocalFeatureExtractor(nn.Module):
    def __init__(self, input_channels=3, output_feature_dim=256):
        super().__init__()
        self.output_dim = output_feature_dim # Store for easy access
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1), # e.g., 3x32x32 -> 16x32x32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # -> 16x16x16
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # -> 32x16x16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # -> 32x8x8
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # -> 64x8x8
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)) # -> 64x1x1
        )
        self.flatten = nn.Flatten() # -> 64
        self.fc = nn.Linear(64, output_feature_dim) # -> output_feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x expected to be image batch, e.g., (batch_size, input_channels, H, W)
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x 

# --- Model Creation Function ---
def create_model(model_type, num_classes, **kwargs):
    """Create model based on type and parameters."""
    if model_type == 'cnn':
        if num_classes == 10:
            if kwargs.get('dataset') == 'mnist':
                return MNISTNet()
            elif kwargs.get('dataset') == 'fashion_mnist':
                return MNISTNet()  # Fashion-MNIST uses same architecture as MNIST
            elif kwargs.get('dataset') == 'svhn':
                return SVHNNet()
            else:  # Default to CIFAR10
                return CIFAR10Net()
        elif num_classes == 100:
            return CIFAR100Net()
        elif num_classes == 2:
            return CelebANet()
    elif model_type == 'mlp':
        if num_classes == 2:
            input_dim = kwargs.get('input_dim', 14)
            if input_dim == 14:
                return AdultNet(input_dim=input_dim)
            elif input_dim == 29:
                return CreditNet(input_dim=input_dim)
        elif num_classes == 5:  # KDD Cup 99
            return KDDNet(input_dim=kwargs.get('input_dim', 41))
        elif num_classes == 7:
            return CovertypeNet(input_dim=kwargs.get('input_dim', 54))
    elif model_type == 'text':
        if num_classes == 4:  # AG News
            vocab_size = kwargs.get('vocab_size', 95811)
            embed_dim = kwargs.get('embed_dim', 100)
            return AGNewsNet(vocab_size=vocab_size, embed_dim=embed_dim, num_class=num_classes)
        elif num_classes == 20:  # 20 Newsgroups
            vocab_size = kwargs.get('vocab_size', 100000)
            embed_dim = kwargs.get('embed_dim', 128)
            return NewsGroupsNet(vocab_size=vocab_size, embed_dim=embed_dim, num_class=num_classes)
    elif model_type == 'lstm':
        vocab_size = kwargs.get('vocab_size', 256)
        return ShakespeareLSTM(vocab_size=vocab_size)
    elif model_type == 'vae':
        input_dim = kwargs.get('input_dim', 784)
        latent_dim = kwargs.get('latent_dim', 20)
        return GenericVAE(input_dim=input_dim, latent_dim=latent_dim)
    
    raise ValueError(f"Unknown model type: {model_type} with {num_classes} classes") 