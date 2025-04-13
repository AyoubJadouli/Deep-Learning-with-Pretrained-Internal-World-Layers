import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_recall_fscore_support, confusion_matrix,
    precision_recall_curve, f1_score, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
from tqdm import tqdm
import gc
warnings.filterwarnings("ignore")

# Environment setup for better stability
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ["TORCHDYNAMO_DISABLE"] = "1"  # Disable torch dynamo to avoid issues

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------------------------------
# MODEL DEFINITIONS 
# -------------------------------------------

# 1. Custom Transformer Layer (to replace Gemma)
class CustomGemmaLayer(nn.Module):
    """
    A custom transformer layer that mimics Gemma's functionality
    without requiring position embeddings.
    """
    def __init__(self, hidden_size=1152, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Self attention
        self.layernorm_before = nn.LayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(
            hidden_size, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # Feed forward
        self.layernorm_after = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
    def forward(self, hidden_states, **kwargs):
        # First residual block: self-attention
        residual = hidden_states
        hidden_states = self.layernorm_before(hidden_states)
        attn_output, _ = self.self_attn(hidden_states, hidden_states, hidden_states)
        hidden_states = residual + attn_output
        
        # Second residual block: feed-forward
        residual = hidden_states
        hidden_states = self.layernorm_after(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


# 2. Gemma3 Internal World Model
class Gemma3InternalWorldModel(nn.Module):
    """
    Multi-branch neural network with a custom internal world layer
    for wildfire prediction.
    """
    def __init__(self, n_features, dropout_rate=0.4, num_internal_layers=2):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = 1152  # Gemma 3-1B hidden dimension
        self.dropout_rate = dropout_rate
        
        # Split features into 4 branches (approximately equal size)
        self.branch_size = n_features // 4
        self.branch_sizes = [self.branch_size] * 3
        self.branch_sizes.append(n_features - sum(self.branch_sizes[:3]))
        
        # Each branch outputs 1152/4 = 288 dimensions
        self.branch_output_dim = self.hidden_dim // 4
        
        # Define the 4 parallel branches
        self.branch1 = self._make_branch(self.branch_sizes[0], self.branch_output_dim)
        self.branch2 = self._make_branch(self.branch_sizes[1], self.branch_output_dim)
        self.branch3 = self._make_branch(self.branch_sizes[2], self.branch_output_dim)
        self.branch4 = self._make_branch(self.branch_sizes[3], self.branch_output_dim)
        
        # 3-layer FFN after concatenation
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim)
        )
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim)
        )
        
        # Use our custom Gemma-like layers (stack multiple for more capacity)
        print(f"Using {num_internal_layers} custom Gemma-like internal layers")
        self.internal_layers = nn.ModuleList([
            CustomGemmaLayer(
                hidden_size=self.hidden_dim,
                num_heads=8,
                dropout=self.dropout_rate
            ) for _ in range(num_internal_layers)
        ])
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 1)
        )
        
        # Initialize weights for better training
        self._initialize_weights()
    
    def _make_branch(self, in_dim, out_dim):
        """Create a feedforward branch with ReLU activation."""
        return nn.Sequential(
            nn.Linear(in_dim, out_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(out_dim // 2),
            nn.Dropout(self.dropout_rate),
            nn.Linear(out_dim // 2, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
            nn.Dropout(self.dropout_rate/2)  # Less dropout at the output
        )
    
    def _initialize_weights(self):
        """Initialize the weights of the trainable layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Only apply xavier to weight matrices (2+ dims), not biases
                if m.weight.dim() > 1:
                    nn.init.xavier_normal_(m.weight, gain=0.7)  # Reduced gain for better stability
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass through the model.
        Input: x - batch of feature vectors (B, n_features)
        Output: Wildfire probability logits (B, 1)
        """
        # Split input into 4 branches
        x1 = x[:, :self.branch_sizes[0]]
        x2 = x[:, self.branch_sizes[0]:self.branch_sizes[0]+self.branch_sizes[1]]
        x3 = x[:, self.branch_sizes[0]+self.branch_sizes[1]:sum(self.branch_sizes[:3])]
        x4 = x[:, sum(self.branch_sizes[:3]):]
        
        # Process each branch
        b1 = self.branch1(x1)
        b2 = self.branch2(x2)
        b3 = self.branch3(x3)
        b4 = self.branch4(x4)
        
        # Concatenate branch outputs to form 1152-dim vector
        concatenated = torch.cat([b1, b2, b3, b4], dim=1)  # (B, 1152)
        
        # Apply 3-layer FFN
        ffn_output = self.ffn(concatenated)
        
        # Apply projection
        projection = self.projection(ffn_output)  # (B, 1152)
        
        # Reshape for internal layer: (B, 1152) -> (B, 1, 1152)
        internal_input = projection.unsqueeze(1)
        
        # Process through each internal layer
        internal_output = internal_input
        for layer in self.internal_layers:
            internal_output = layer(internal_output)
        
        # Squeeze sequence dimension: (B, 1, 1152) -> (B, 1152)
        internal_output = internal_output.squeeze(1)
        
        # Classification head
        output = self.classifier(internal_output)
        
        # Return logits directly (don't apply sigmoid)
        return output


# 3. Custom Entropy Layer
class EntropyLayer(nn.Module):
    def __init__(self, n_landcover, m_env_factors):
        super(EntropyLayer, self).__init__()
        # Trainable scaling constant for entropy term
        self.k = nn.Parameter(torch.ones(1))
        # Trainable weights for environmental factors
        self.alpha = nn.Parameter(torch.ones(m_env_factors))
        self.n_landcover = n_landcover
        self.m_env_factors = m_env_factors

    def forward(self, inputs):
        # Get the actual dimensions of the input
        input_shape = inputs.size()
        n_features = input_shape[1]
        
        # Adjust n_landcover if it's larger than the input size
        n_landcover_adjusted = min(self.n_landcover, n_features)
        
        # Split input into land cover proportions and environmental factors
        p_i = F.softmax(inputs[:, :n_landcover_adjusted], dim=-1)
        f_j = inputs[:, n_landcover_adjusted:]
        
        # Get the actual size of f_j
        f_j_size = f_j.size(1)
        
        # Use only as many alpha values as there are features in f_j
        alpha_adjusted = self.alpha[:f_j_size]
        
        # Calculate entropy term (landscape diversity)
        entropy_term = -self.k * torch.sum(
                    p_i * torch.log(p_i + 1e-10), dim=-1)
        
        # Calculate environmental influence term
        env_term = torch.sum(alpha_adjusted * f_j, dim=-1)
        
        # Return combined entropy score (scalar per sample)
        return (entropy_term + env_term).unsqueeze(1)


# 4. Physics-Embedded Entropy Model (Full Model)
class FullEntropyModel(nn.Module):
    def __init__(self, n_features, n_landcover, m_env_factors):
        super(FullEntropyModel, self).__init__()
        # FFN Branch
        self.ffn_branch1 = nn.Linear(n_features, 256)
        self.ffn_bn1 = nn.BatchNorm1d(256)
        self.ffn_dropout = nn.Dropout(0.3)
        self.ffn_branch2 = nn.Linear(256, 128)
        self.ffn_bn2 = nn.BatchNorm1d(128)
        
        # 1D CNN Branch
        self.cnn_branch = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.cnn_bn = nn.BatchNorm1d(32)
        self.cnn_dense = nn.Linear(32 * n_features, 128)
        self.cnn_bn2 = nn.BatchNorm1d(128)
        
        # PMFFNN Branch
        self.split_size = n_features // 3
        # Branch 1
        self.pmffnn_branch1 = nn.Linear(self.split_size, 64)
        self.pmffnn_bn1 = nn.BatchNorm1d(64)
        
        # Branch 2
        self.pmffnn_branch2 = nn.Linear(self.split_size, 64)
        self.pmffnn_bn2 = nn.BatchNorm1d(64)
        
        # Branch 3 (might be slightly larger if n_features isn't divisible by 3)
        self.pmffnn_branch3 = nn.Linear(n_features - 2*self.split_size, 64)
        self.pmffnn_bn3 = nn.BatchNorm1d(64)
        
        # PMFFNN integration
        self.pmffnn_dense = nn.Linear(64 * 3, 128)
        self.pmffnn_bn4 = nn.BatchNorm1d(128)
        
        # Integration Network
        self.integrated_dense1 = nn.Linear(128 * 3, 512)
        self.integrated_bn1 = nn.BatchNorm1d(512)
        self.integrated_dropout = nn.Dropout(0.3)
        self.integrated_dense2 = nn.Linear(512, 256)
        self.integrated_bn2 = nn.BatchNorm1d(256)
        
        # Physics-Embedded Entropy Layer
        self.entropy_layer = EntropyLayer(n_landcover, m_env_factors)
        
        # Multi-path classification with sigmoid layers
        self.sigmoid_branch1 = nn.Linear(128 + 1, 128)
        self.sigmoid_branch2 = nn.Linear(128 + 1, 128)
        self.sigmoid_branch3 = nn.Linear(128 + 1, 128)
        
        # Output layer
        self.output_layer = nn.Linear(128 * 3, 1)
        
    def forward(self, x):
        # FFN Branch
        ffn = F.gelu(self.ffn_branch1(x))
        ffn = self.ffn_bn1(ffn)
        ffn = self.ffn_dropout(ffn)
        ffn = F.gelu(self.ffn_branch2(ffn))
        ffn_out = self.ffn_bn2(ffn)
        
        # 1D CNN Branch
        cnn_input = x.unsqueeze(1)  # Reshape for CNN input
        cnn = F.selu(self.cnn_branch(cnn_input))
        cnn = self.cnn_bn(cnn)
        cnn = cnn.view(cnn.size(0), -1)  # Flatten
        cnn = F.selu(self.cnn_dense(cnn))
        cnn_out = self.cnn_bn2(cnn)
        
        # PMFFNN Branch - split features into 3 groups
        branch1_input = x[:, :self.split_size]
        branch2_input = x[:, self.split_size:2*self.split_size]
        branch3_input = x[:, 2*self.split_size:]
        
        branch1 = F.selu(self.pmffnn_branch1(branch1_input))
        branch1 = self.pmffnn_bn1(branch1)
        
        branch2 = F.selu(self.pmffnn_branch2(branch2_input))
        branch2 = self.pmffnn_bn2(branch2)
        
        branch3 = F.selu(self.pmffnn_branch3(branch3_input))
        branch3 = self.pmffnn_bn3(branch3)
        
        # Concatenate PMFFNN branches
        pmffnn_concat = torch.cat([branch1, branch2, branch3], dim=1)
        pmffnn = F.selu(self.pmffnn_dense(pmffnn_concat))
        pmffnn_out = self.pmffnn_bn4(pmffnn)
        
        # Concatenate all branch outputs
        concat = torch.cat([ffn_out, cnn_out, pmffnn_out], dim=1)
        
        # Integration Network
        integrated = F.gelu(self.integrated_dense1(concat))
        integrated = self.integrated_bn1(integrated)
        integrated = self.integrated_dropout(integrated)
        integrated = F.gelu(self.integrated_dense2(integrated))
        integrated = self.integrated_bn2(integrated)
        
        # Physics-Embedded Entropy Layer
        entropy_out = self.entropy_layer(integrated)
        
        # Residual connection from FFN branch
        combined = torch.cat([entropy_out, ffn_out], dim=1)
        
        # Multi-path classification with sigmoid layers
        sig_branch1 = torch.sigmoid(self.sigmoid_branch1(combined))
        sig_branch2 = torch.sigmoid(self.sigmoid_branch2(combined))
        sig_branch3 = torch.sigmoid(self.sigmoid_branch3(combined))
        
        # Concatenate sigmoid branches
        sig_concat = torch.cat([sig_branch1, sig_branch2, sig_branch3], dim=1)
        
        # Output
        output = self.output_layer(sig_concat)
        
        return output


# 5. FFN-only Model for comparison
class FFNModel(nn.Module):
    def __init__(self, n_features):
        super(FFNModel, self).__init__()
        self.dense1 = nn.Linear(n_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        
        self.dense2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        
        self.dense3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.output_layer = nn.Linear(64, 1)
    
    def forward(self, x):
        x = F.gelu(self.dense1(x))
        x = self.bn1(x)
        x = self.dropout1(x)
        
        x = F.gelu(self.dense2(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        
        x = F.gelu(self.dense3(x))
        x = self.bn3(x)
        
        x = self.output_layer(x)
        return x


# 6. 1D CNN-only Model for comparison
class CNNModel(nn.Module):
    def __init__(self, n_features):
        super(CNNModel, self).__init__()
        # Reshape for CNN
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        
        # Flatten and dense layers
        self.dense1 = nn.Linear(64 * n_features, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.3)
        
        self.output_layer = nn.Linear(128, 1)
    
    def forward(self, x):
        # Reshape input for CNN
        x = x.unsqueeze(1)  # Add channel dimension
        
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.dense1(x))
        x = self.bn3(x)
        x = self.dropout(x)
        
        x = self.output_layer(x)
        return x


# 7. FFN with Positional Encoding model
class FFNWithPosEncoding(nn.Module):
    def __init__(self, n_features):
        super(FFNWithPosEncoding, self).__init__()
        self.embed_dim = 32  # Embedding size for each token
        self.ff_dim = 32  # Hidden layer size in feed forward network
        
        # Feed forward network
        self.ffn = nn.Sequential(
            nn.Linear(1, self.ff_dim),
            nn.ReLU(),
            nn.Linear(self.ff_dim, self.embed_dim)
        )
        
        # Layer normalization
        self.layernorm = nn.LayerNorm(self.embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        # Positional embedding
        self.pos_emb = nn.Embedding(n_features, self.embed_dim)
        
        # Register buffer for positions
        positions = torch.arange(0, n_features).long()
        self.register_buffer('positions', positions)
        
        # Output layer
        self.output_layer = nn.Linear(n_features * self.embed_dim, 1)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Expand dimensions of inputs to match embedding output
        x = x.unsqueeze(2)  # Shape becomes [batch_size, n_features, 1]
        
        # Pass inputs through the feed forward network
        x = self.ffn(x)  # Shape becomes [batch_size, n_features, embed_dim]
        
        # Get positional embeddings
        pos_encoding = self.pos_emb(self.positions)  # Shape [n_features, embed_dim]
        
        # Expand pos_encoding to match the batch size of inputs
        pos_encoding = pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Add dropout
        x = self.dropout(x)
        
        # Add positional encodings
        x = x + pos_encoding
        
        # Apply layer normalization
        x = self.layernorm(x)
        
        # Flatten the output for the classifier
        x = x.reshape(batch_size, -1)
        
        # Output
        x = self.output_layer(x)
        return x


# -------------------------------------------
# TRAINING AND EVALUATION FUNCTIONS
# -------------------------------------------

def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None, threshold=0.5):
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_outputs = []
    
    for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        
        # Backward pass and optimize
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Step the scheduler if it exists
        if scheduler is not None:
            scheduler.step()
        
        # Track statistics
        total_loss += loss.item() * inputs.size(0)
        total += targets.size(0)
        preds = (torch.sigmoid(logits) > threshold).float()
        correct += (preds == targets).sum().item()
        
        # Store for AUC calculation
        all_targets.extend(targets.cpu().numpy().ravel())
        all_outputs.extend(torch.sigmoid(logits).detach().cpu().numpy().ravel())
    
    # Calculate epoch metrics
    epoch_loss = total_loss / total
    epoch_acc = correct / total
    epoch_auc = roc_auc_score(all_targets, all_outputs)
    
    return epoch_loss, epoch_acc, epoch_auc, all_targets, all_outputs


def find_optimal_threshold(targets, outputs):
    """Find the optimal threshold for classification based on F1 score."""
    precisions, recalls, thresholds = precision_recall_curve(targets, outputs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    
    # Find best threshold
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    return best_threshold


def evaluate(model, dataloader, criterion, device, threshold=0.5, find_best_threshold=False):
    """Evaluate model on validation data."""
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            logits = model(inputs)
            loss = criterion(logits, targets)
            
            # Track statistics
            total_loss += loss.item() * inputs.size(0)
            
            # Store for metrics calculation
            all_targets.extend(targets.cpu().numpy().ravel())
            all_outputs.extend(torch.sigmoid(logits).cpu().numpy().ravel())
    
    # Find best threshold if requested
    if find_best_threshold:
        threshold = find_optimal_threshold(all_targets, all_outputs)
        print(f"Optimal threshold: {threshold:.4f}")
    
    # Apply threshold to get predictions
    all_preds = (np.array(all_outputs) > threshold).astype(float)
    
    # Calculate metrics
    epoch_loss = total_loss / len(all_targets)
    epoch_acc = (all_preds == all_targets).mean()
    epoch_auc = roc_auc_score(all_targets, all_outputs)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary')
    cm = confusion_matrix(all_targets, all_preds)
    
    return epoch_loss, epoch_acc, epoch_auc, precision, recall, f1, cm, threshold, all_targets, all_outputs


# -------------------------------------------
# DATA LOADING AND PREPARATION
# -------------------------------------------

def load_morocco_wildfire_data(data_path="../Data/Data/FinalDataSet/Date_final_dataset_balanced_float32.parquet"):
    """Load the Morocco wildfire dataset or create simulated data if not available."""
    try:
        df = pd.read_parquet(data_path)
        print(f"Dataset shape: {df.shape}")
    except FileNotFoundError:
        print(f"File {data_path} not found. Creating simulated data...")
        # Create simulated data with appropriate structure
        n_samples = 10000
        n_features = 276  # Matching the real dataset feature count
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = (np.random.rand(n_samples) > 0.5).astype(np.float32)
        dates = pd.date_range(start='2010-01-01', end='2022-12-31', periods=n_samples)
        
        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['is_fire'] = y
        df['acq_date'] = dates
        print(f"Created simulated dataset with shape: {df.shape}")
    
    return df


def prepare_data(df, batch_size=128, sample_size=None):
    """Split data, balance classes, standardize, and create data loaders."""
    print("Performing time-based train/validation split...")
    df_train = df[df.acq_date < '2022-01-01']
    df_valid = df[df.acq_date >= '2022-01-01']
    
    print(f"Training set (before balancing): {df_train.shape}")
    print(f"Validation set (before balancing): {df_valid.shape}")
    
    # Balance datasets by class
    print("Balancing datasets...")
    min_samples_train = df_train['is_fire'].value_counts().min()
    min_samples_valid = df_valid['is_fire'].value_counts().min()
    
    # If sample_size is provided, limit the number of samples per class
    if sample_size and sample_size < min_samples_train:
        min_samples_train = sample_size
    if sample_size and sample_size < min_samples_valid:
        min_samples_valid = sample_size
        
    # Balance the training dataset
    df_train_balanced = pd.concat([
        df_train[df_train['is_fire'] == 0].sample(min_samples_train, random_state=42),
        df_train[df_train['is_fire'] == 1].sample(min_samples_train, random_state=42)
    ])
    
    # Balance the validation dataset
    df_valid_balanced = pd.concat([
        df_valid[df_valid['is_fire'] == 0].sample(min_samples_valid, random_state=42),
        df_valid[df_valid['is_fire'] == 1].sample(min_samples_valid, random_state=42)
    ])
    
    # Shuffle both datasets
    df_train_balanced = df_train_balanced.sample(frac=1, random_state=42)
    df_valid_balanced = df_valid_balanced.sample(frac=1, random_state=42)
    
    print(f"Balanced training set: {df_train_balanced.shape}")
    print(f"Balanced validation set: {df_valid_balanced.shape}")
    
    # Remove acquisition date column
    print("Removing acquisition date column...")
    if 'acq_date' in df_train_balanced.columns:
        df_train_balanced = df_train_balanced.drop(columns=['acq_date'])
        df_valid_balanced = df_valid_balanced.drop(columns=['acq_date'])
    
    # Extract features and target
    print("Preparing feature sets...")
    y_train = df_train_balanced['is_fire']
    X_train = df_train_balanced.drop(columns=['is_fire'])
    
    y_valid = df_valid_balanced['is_fire']
    X_valid = df_valid_balanced.drop(columns=['is_fire'])
    
    # Standardize features
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    
    n_features = X_train_scaled.shape[1]
    print(f"Total number of features: {n_features}")
    
    # Create tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1)
    X_valid_tensor = torch.FloatTensor(X_valid_scaled)
    y_valid_tensor = torch.FloatTensor(y_valid.values).unsqueeze(1)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, n_features


# -------------------------------------------
# MODEL TRAINING FUNCTION
# -------------------------------------------

def train_and_evaluate_model(model, model_name, train_loader, valid_loader, device, num_epochs=5, lr=0.001, 
                           weight_decay=0.01, use_scheduler=True, pos_weight=None):
    """Train and evaluate a model."""
    print(f"\n{'='*80}")
    print(f"===== Training {model_name} =====")
    print(f"{'='*80}")
    
    # Move model to device
    model = model.to(device)
    
    # Define loss - use pos_weight if provided
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Create scheduler if requested
    if use_scheduler:
        total_steps = len(train_loader) * num_epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )
    else:
        scheduler = None
    
    # For storing history
    history = {
        'train_loss': [], 'train_acc': [], 'train_auc': [],
        'val_loss': [], 'val_acc': [], 'val_auc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': [],
        'threshold': []
    }
    
    # For early stopping
    best_val_f1 = 0.0
    best_epoch = -1
    best_threshold = 0.5
    best_model_path = f"{model_name.replace(' ', '_').lower()}_best_model.pt"
    patience = 3
    patience_counter = 0
    
    # Record start time
    start_time = time.time()
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        train_loss, train_acc, train_auc, _, _ = train_epoch(
            model, train_loader, criterion, optimizer, device, scheduler, best_threshold
        )
        
        # Validation phase (finding best threshold every other epoch)
        find_best_threshold = (epoch % 2 == 0) or (epoch == num_epochs - 1)
        val_loss, val_acc, val_auc, val_precision, val_recall, val_f1, cm, threshold, _, _ = evaluate(
            model, valid_loader, criterion, device, 
            threshold=best_threshold, find_best_threshold=find_best_threshold
        )
        
        if find_best_threshold:
            best_threshold = threshold
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_auc'].append(train_auc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        history['threshold'].append(threshold)
        
        # Print progress
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Training:   loss={train_loss:.4f}, acc={train_acc:.4f}, auc={train_auc:.4f}")
        print(f"  Validation: loss={val_loss:.4f}, acc={val_acc:.4f}, auc={val_auc:.4f}")
        print(f"             precision={val_precision:.4f}, recall={val_recall:.4f}, f1={val_f1:.4f}")
        print(f"             threshold={threshold:.4f}")
        
        # Save best model based on F1 score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved! (F1={val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs. Best F1={best_val_f1:.4f} at epoch {best_epoch+1}")
            
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Calculate training time
    training_time = time.time() - start_time
    print(f"\n{model_name} training time: {training_time:.2f} seconds")
    
    # Load best model for final evaluation
    if best_epoch >= 0:
        print(f"Loading best model from epoch {best_epoch+1}...")
        model.load_state_dict(torch.load(best_model_path))
    
    # Final evaluation with optimal threshold
    val_loss, val_acc, val_auc, val_precision, val_recall, val_f1, cm, _, val_targets, val_outputs = evaluate(
        model, valid_loader, criterion, device, threshold=best_threshold, find_best_threshold=False
    )
    
    # Print final results
    print(f"\n{model_name} Final Results:")
    print(f"  Test Accuracy: {val_acc:.4f}")
    print(f"  Test AUC: {val_auc:.4f}")
    print(f"  Test Precision: {val_precision:.4f}")
    print(f"  Test Recall: {val_recall:.4f}")
    print(f"  Test F1 Score: {val_f1:.4f}")
    print(f"  Optimal Threshold: {best_threshold:.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {trainable_params:,} trainable out of {total_params:,} total")
    
    # Return results
    return {
        'model': model,
        'model_name': model_name,
        'history': history,
        'training_time': training_time,
        'test_loss': val_loss,
        'test_acc': val_acc,
        'test_auc': val_auc,
        'precision': val_precision,
        'recall': val_recall,
        'f1': val_f1,
        'confusion_matrix': cm,
        'best_threshold': best_threshold,
        'best_epoch': best_epoch,
        'val_targets': val_targets,
        'val_outputs': val_outputs,
        'total_params': total_params,
        'trainable_params': trainable_params
    }


# -------------------------------------------
# VISUALIZATION FUNCTIONS
# -------------------------------------------

def plot_confusion_matrix(cm, model_name):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Fire', 'Fire'],
                yticklabels=['No Fire', 'Fire'])
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{model_name.replace(' ', '_').lower()}_confusion_matrix.png")
    return plt.gcf()


def plot_roc_curve(results_list):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(10, 8))
    
    for result in results_list:
        fpr, tpr, _ = roc_curve(result['val_targets'], result['val_outputs'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, 
                 label=f"{result['model_name']} (AUC = {roc_auc:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("roc_curves_comparison.png")
    return plt.gcf()


def plot_precision_recall_curve(results_list):
    """Plot precision-recall curves for all models."""
    plt.figure(figsize=(10, 8))
    
    for result in results_list:
        precision, recall, _ = precision_recall_curve(result['val_targets'], result['val_outputs'])
        avg_precision = np.mean(precision)
        plt.plot(recall, precision, lw=2, 
                 label=f"{result['model_name']} (Avg Precision = {avg_precision:.3f})")
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves', fontsize=14)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("precision_recall_curves_comparison.png")
    return plt.gcf()


def plot_training_curves(history, model_name):
    """Plot training and validation curves."""
    plt.figure(figsize=(16, 12))
    
    # Loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy Curves', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # AUC
    plt.subplot(2, 2, 3)
    plt.plot(history['train_auc'], label='Train AUC')
    plt.plot(history['val_auc'], label='Validation AUC')
    plt.title('AUC Curves', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('AUC', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Precision, Recall, F1
    plt.subplot(2, 2, 4)
    plt.plot(history['val_precision'], label='Precision')
    plt.plot(history['val_recall'], label='Recall')
    plt.plot(history['val_f1'], label='F1 Score')
    plt.title('Precision, Recall, and F1 Score', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'Training Curves - {model_name}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(f"{model_name.replace(' ', '_').lower()}_training_curves.png")
    return plt.gcf()


def compare_metrics(results_list):
    """Create comparison plots for multiple models."""
    # Comparison of metrics in a bar chart
    metrics = ['test_acc', 'test_auc', 'precision', 'recall', 'f1']
    metric_names = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score']
    
    plt.figure(figsize=(14, 8))
    x = np.arange(len(metric_names))
    width = 0.8 / len(results_list)  # Adjust bar width based on number of models
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_list)))
    
    for i, result in enumerate(results_list):
        values = [result[metric] for metric in metrics]
        offset = width * (i - (len(results_list) - 1) / 2)
        bars = plt.bar(x + offset, values, width, label=result['model_name'], color=colors[i])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', rotation=90, fontsize=9)
    
    plt.title('Performance Metrics Comparison', fontsize=16)
    plt.xticks(x, metric_names, fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('metrics_comparison.png')
    
    # Training time comparison
    plt.figure(figsize=(12, 6))
    model_names = [r['model_name'] for r in results_list]
    training_times = [r['training_time'] for r in results_list]
    
    bars = plt.bar(model_names, training_times, color=colors)
    plt.title('Training Time Comparison', fontsize=16)
    plt.ylabel('Time (seconds)', fontsize=14)
    plt.xticks(fontsize=12, rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add time values on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}s', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('training_time_comparison.png')
    
    # Parameter count comparison
    plt.figure(figsize=(12, 6))
    model_names = [r['model_name'] for r in results_list]
    param_counts = [r['trainable_params'] for r in results_list]
    
    bars = plt.bar(model_names, param_counts, color=colors)
    plt.title('Model Parameter Count Comparison', fontsize=16)
    plt.ylabel('Number of Parameters', fontsize=14)
    plt.xticks(fontsize=12, rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.yscale('log')
    
    # Add parameter count on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                f'{height:,.0f}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('parameter_count_comparison.png')
    
    return None


def visualize_architecture(model_name):
    """Create ASCII diagram of model architecture."""
    if model_name == "Internal World Model":
        diagram = """
        +--------------------+
        | Input Features     |
        +----------+---------+
                   |
        +----------v---------+
        |   4 Parallel       |
        |  Feature Branches  |
        +----------+---------+
                   |
        +----------v---------+
        |    Concatenate     |
        |   (1152-dim)       |
        +----------+---------+
                   |
        +----------v---------+
        |  3-Layer FFN       |
        +----------+---------+
                   |
        +----------v---------+
        | Projection Layer   |
        +----------+---------+
                   |
        +----------v---------+
        |  Internal World    |
        | (Transformer Layers)|
        +----------+---------+
                   |
        +----------v---------+
        | Classification MLP |
        +----------+---------+
                   |
        +----------v---------+
        |    Fire Prediction |
        +--------------------+
        """
    elif model_name == "Physics-Embedded Entropy Model":
        diagram = """
        +--------------------+
        | Input Features     |
        +----------+---------+
                   |
            +------+------+------+
            |             |      |
        +---v----+    +---v----+ +---v----+
        |  FFN   |    |  CNN   | |  PMFFN  |
        | Branch |    | Branch | | Branch  |
        +---+----+    +---+----+ +---+----+
            |             |          |
            +------+------+----------+
                   |
        +----------v---------+
        | Integration Network|
        +----------+---------+
                   |
        +----------v---------+
        |  Entropy Layer     |
        | (Physics-Informed) |
        +----------+---------+
                   |
        +----------v---------+
        | Multi-path Sigmoid |
        |    Classification  |
        +----------+---------+
                   |
        +----------v---------+
        |    Fire Prediction |
        +--------------------+
        """
    elif model_name == "FFN Model":
        diagram = """
        +--------------------+
        | Input Features     |
        +----------+---------+
                   |
        +----------v---------+
        |  Dense Layer 1     |
        |  (256 units + BN)  |
        +----------+---------+
                   |
        +----------v---------+
        |  Dense Layer 2     |
        |  (128 units + BN)  |
        +----------+---------+
                   |
        +----------v---------+
        |  Dense Layer 3     |
        |  (64 units + BN)   |
        +----------+---------+
                   |
        +----------v---------+
        |    Fire Prediction |
        +--------------------+
        """
    elif model_name == "CNN Model":
        diagram = """
        +--------------------+
        | Input Features     |
        +----------+---------+
                   |
        +----------v---------+
        | Reshape to 1D      |
        +----------+---------+
                   |
        +----------v---------+
        | Conv1D Layer 1     |
        | (32 filters)       |
        +----------+---------+
                   |
        +----------v---------+
        | Conv1D Layer 2     |
        | (64 filters)       |
        +----------+---------+
                   |
        +----------v---------+
        | Flatten + Dense    |
        | (128 units)        |
        +----------+---------+
                   |
        +----------v---------+
        |    Fire Prediction |
        +--------------------+
        """
    elif model_name == "FFN with Positional Encoding":
        diagram = """
        +--------------------+
        | Input Features     |
        +----------+---------+
                   |
        +----------v---------+
        | Reshape Each Feature|
        +----------+---------+
                   |
        +----------v---------+
        | Feature-wise FFN   |
        +----------+---------+
                   |
        +----------v---------+
        | Add Positional     |
        | Encodings          |
        +----------+---------+
                   |
        +----------v---------+
        | LayerNorm + Dropout|
        +----------+---------+
                   |
        +----------v---------+
        | Flatten + Dense    |
        +----------+---------+
                   |
        +----------v---------+
        |    Fire Prediction |
        +--------------------+
        """
    else:
        diagram = "Architecture diagram not available."
    
    # Return the diagram
    return diagram


# -------------------------------------------
# MAIN EXECUTION
# -------------------------------------------

print("=" * 80)
print("===== MOROCCO WILDFIRE PREDICTION BENCHMARK =====")
print("=" * 80)

# Set parameters
BATCH_SIZE = 128
NUM_EPOCHS = 5
SAMPLE_SIZE = None  # Set to a number (e.g., 10000) for faster testing or None for full dataset

# Load and prepare data
data_path = "../Data/Data/FinalDataSet/Date_final_dataset_balanced_float32.parquet"
df = load_morocco_wildfire_data(data_path)
train_loader, valid_loader, n_features = prepare_data(df, batch_size=BATCH_SIZE, sample_size=SAMPLE_SIZE)

# Show feature information
print(f"\nTraining with {n_features} features")

# For entropy layer parameters
n_landcover = 4  # For entropy layer - represents NDVI-related features
m_env_factors = min(300, n_features - n_landcover)  # Environmental factors

# Create all models
print("\nInitializing models...")

# Create the Internal World model
internal_world_model = Gemma3InternalWorldModel(
    n_features=n_features, 
    dropout_rate=0.4,
    num_internal_layers=2
)

# Create the Physics-Embedded Entropy model
entropy_model = FullEntropyModel(
    n_features=n_features,
    n_landcover=n_landcover,
    m_env_factors=m_env_factors
)

# Create the FFN model
ffn_model = FFNModel(n_features=n_features)

# Create the CNN model
cnn_model = CNNModel(n_features=n_features)

# Create the FFN with Positional Encoding model
ffn_pos_model = FFNWithPosEncoding(n_features=n_features)

# Create a list of models and their names to train
models = [
    (internal_world_model, "Internal World Model"),
    (entropy_model, "Physics-Embedded Entropy Model"),
    (ffn_model, "FFN Model"),
    (cnn_model, "CNN Model"),
    (ffn_pos_model, "FFN with Positional Encoding")
]

# Train all models and collect results
results_list = []
for model, model_name in models:
    # Print model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{model_name} has {trainable_params:,} trainable parameters out of {total_params:,} total")
    
    # Train and evaluate the model
    result = train_and_evaluate_model(
        model=model,
        model_name=model_name,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        num_epochs=NUM_EPOCHS,
        lr=0.001,
        weight_decay=0.01,
        use_scheduler=True,
        pos_weight=2.0  # Give higher weight to positive class (fire events)
    )
    
    # Plot confusion matrix
    plot_confusion_matrix(result['confusion_matrix'], model_name)
    
    # Plot training curves
    plot_training_curves(result['history'], model_name)
    
    # Visualize model architecture
    architecture_diagram = visualize_architecture(model_name)
    print(f"\nArchitecture of {model_name}:")
    print(architecture_diagram)
    
    # Collect results for comparison
    results_list.append(result)
    
    # Clear memory
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Compare all models
print("\n" + "="*80)
print("===== MODEL COMPARISON =====")
print("="*80 + "\n")

# Create a comprehensive comparison table
comparison_data = {
    'Model': [r['model_name'] for r in results_list],
    'Accuracy': [r['test_acc'] for r in results_list],
    'AUC': [r['test_auc'] for r in results_list],
    'Precision': [r['precision'] for r in results_list],
    'Recall': [r['recall'] for r in results_list],
    'F1 Score': [r['f1'] for r in results_list],
    'Training Time (s)': [r['training_time'] for r in results_list],
    'Parameters': [r['trainable_params'] for r in results_list]
}

# Create a DataFrame for better visualization
comparison_df = pd.DataFrame(comparison_data)

# Format the numeric columns
for col in comparison_df.columns:
    if col not in ['Model', 'Parameters']:
        comparison_df[col] = comparison_df[col].map(lambda x: f"{x:.4f}")

# Format the parameters column
comparison_df['Parameters'] = comparison_df['Parameters'].map(lambda x: f"{x:,}")

print(comparison_df.to_string(index=False))

# Save the comparison DataFrame to CSV
comparison_df.to_csv('model_comparison.csv', index=False)

# Plot ROC and Precision-Recall curves
plot_roc_curve(results_list)
plot_precision_recall_curve(results_list)

# Visualize comparison metrics
compare_metrics(results_list)

print("\nAll visualizations saved. Benchmark complete!")


