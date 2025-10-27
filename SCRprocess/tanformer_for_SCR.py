import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1️⃣ 读取数据
file_path = "tuox_hd_20250710_0722.csv"
df = pd.read_csv(file_path, sep='\t')  # 如果是逗号分隔就改为 sep=','

# 删除第一列 'date'
df = df.drop(columns=['date'])

# 输入特征和输出
X = df[['in-o2', '1-valve', 'in-nox']].values
y = df[['out-nox']].values

# 标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# 分割训练测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2️⃣ 数据集定义
class NOxDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Transformer 期望输入形状: (seq_len, batch, feature_dim)
        # 这里每个样本看作长度为1的序列
        return self.X[idx].unsqueeze(0), self.y[idx]

train_dataset = NOxDataset(X_train, y_train)
test_dataset = NOxDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# 3️⃣ 定义 Transformer 模型
class TransformerRegressor(nn.Module):
    def __init__(self, input_dim=3, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_linear = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch, seq_len, feature_dim)
        x = self.input_linear(x)                # (batch, seq_len, d_model)
        x = x.permute(1, 0, 2)                  # 转换为 (seq_len, batch, d_model)
        x = self.transformer(x)                 # Transformer 编码
        x = x.mean(dim=0)                       # 池化 (取平均)
        out = self.output_linear(x)             # 输出预测
        return out

# 4️⃣ 训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerRegressor().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/100] | Loss: {total_loss/len(train_loader):.6f}")

# 5️⃣ 测试
model.eval()
with torch.no_grad():
    preds = []
    trues = []
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_pred = model(X_batch)
        preds.append(y_pred.cpu())
        trues.append(y_batch)
    preds = torch.cat(preds)
    trues = torch.cat(trues)

# 反标准化
preds_real = scaler_y.inverse_transform(preds)
trues_real = scaler_y.inverse_transform(trues)
print("✅ 测试集预测完成")
