import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class Standardizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std[self.std == 0.0] = 1.0

    def transform(self, data):
        return (data - self.mean) / self.std

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        return data * self.std + self.mean

def generate_gaussian_dataset(num_samples=12000, sequence_length=12):
    true_means = np.random.uniform(-5.0, 5.0, size=(num_samples, 1))
    true_vars = np.random.uniform(0.5, 9.0, size=(num_samples, 1))
    true_stds = np.sqrt(true_vars)
    Z = np.random.normal(0.0, 1.0, size=(num_samples, sequence_length))
    X = true_means + true_stds * Z
    Y = np.hstack([true_means, true_vars])
    return X, Y

# Generate dataset
X_raw, Y_raw = generate_gaussian_dataset()
n_total = len(X_raw)
n_train = int(0.70 * n_total)
n_val = int(0.15 * n_total)

X_train_raw, Y_train_raw = X_raw[:n_train], Y_raw[:n_train]
X_val_raw, Y_val_raw = X_raw[n_train:n_train+n_val], Y_raw[n_train:n_train+n_val]
X_test_raw, Y_test_raw = X_raw[n_train+n_val:], Y_raw[n_train+n_val:]

input_scaler = Standardizer()
target_scaler = Standardizer()

X_train = input_scaler.fit_transform(X_train_raw)
Y_train = target_scaler.fit_transform(Y_train_raw)

X_val = input_scaler.transform(X_val_raw)
Y_val = target_scaler.transform(Y_val_raw)

X_test = input_scaler.transform(X_test_raw)
Y_test = target_scaler.transform(Y_test_raw)

# PyTorch dataset and loaders
class GaussianDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

train_loader = DataLoader(GaussianDataset(X_train, Y_train), batch_size=128, shuffle=True)

# Build Production model to analyze
class ProductionResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.15):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.ln2(out)
        return self.act(out + residual)

class ProductionNet(nn.Module):
    def __init__(self, input_dim=12, hidden_dims=[64, 64, 64], output_dim=2, dropout=0.15):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dims[0]), nn.GELU()]
        for i in range(len(hidden_dims) - 1):
            if hidden_dims[i] == hidden_dims[i+1]:
                layers.append(ProductionResidualBlock(hidden_dims[i], dropout))
            else:
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                layers.append(nn.GELU())
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        return self.head(self.backbone(x))

model = ProductionNet()
optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
criterion = nn.MSELoss()

# We will track standard deviation R2 score and MSE on the validation set at every epoch
epochs = 150
std_r2s = []
std_mses = []
var_r2s = []
var_mses = []

X_val_t = torch.tensor(X_val, dtype=torch.float32)
val_Y_unscaled = target_scaler.inverse_transform(Y_val)
val_true_var = val_Y_unscaled[:, 1]
val_true_std = np.sqrt(val_true_var)

for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

    # Evaluation on val
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_val_t).numpy()
    preds_unscaled = target_scaler.inverse_transform(preds_scaled)
    pred_var = preds_unscaled[:, 1]

    # Compute std: clamp predicted variance to >= 0 to avoid NaNs
    pred_std = np.sqrt(np.maximum(0.0, pred_var))

    # Compute metrics
    r2_std = r2_score(val_true_std, pred_std)
    mse_std = mean_squared_error(val_true_std, pred_std)
    r2_var = r2_score(val_true_var, pred_var)
    mse_var = mean_squared_error(val_true_var, pred_var)

    std_r2s.append(r2_std)
    std_mses.append(mse_std)
    var_r2s.append(r2_var)
    var_mses.append(mse_var)

# Let's plot standard deviation prediction metrics over epochs
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(std_r2s, label='Standard Deviation R² Score', color='tab:orange', lw=2)
plt.plot(var_r2s, label='Variance R² Score', color='tab:blue', linestyle='--', alpha=0.7)
plt.title('R² Score Improvement over Epochs')
plt.xlabel('Epoch')
plt.ylabel('R² Score')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(std_mses, label='Standard Deviation MSE', color='tab:orange', lw=2)
plt.plot(var_mses, label='Variance MSE', color='tab:blue', linestyle='--', alpha=0.7)
plt.title('MSE Reduction over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('std_prediction_analysis.png', dpi=150)
plt.close()

# Let's generate a scatter plot of True vs Predicted Standard Deviation
# for the test set at the end of training
X_test_t = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    test_preds_scaled = model(X_test_t).numpy()
test_preds_unscaled = target_scaler.inverse_transform(test_preds_scaled)
test_true_unscaled = target_scaler.inverse_transform(Y_test)

test_true_var = test_true_unscaled[:, 1]
test_true_std = np.sqrt(test_true_var)

test_pred_var = test_preds_unscaled[:, 1]
test_pred_std = np.sqrt(np.maximum(0.0, test_pred_var))

r2_std_final = r2_score(test_true_std, test_pred_std)

plt.figure(figsize=(7, 6))
plt.scatter(test_true_std, test_pred_std, alpha=0.3, color='tab:orange')
min_val = min(test_true_std.min(), test_pred_std.min())
max_val = max(test_true_std.max(), test_pred_std.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Identity (Perfect Fit)')
plt.title(f'True vs Predicted Standard Deviation (R²: {r2_std_final:.4f})')
plt.xlabel('True Standard Deviation')
plt.ylabel('Predicted Standard Deviation')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('prediction_scatters_with_std.png', dpi=150)
plt.close()

print("Analysis run successfully!")
print(f"Final Standard Deviation R^2 score: {r2_std_final:.4f}")
print(f"Initial Standard Deviation R^2 score: {std_r2s[0]:.4f}")
