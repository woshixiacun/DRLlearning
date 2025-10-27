import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
# --------------------------------------------------
# 1️⃣ 读取 + 预处理
# --------------------------------------------------
def load_and_preprocess(csv_path: str,
                        test_size: float = 0.2,
                        random_state: int = 42):
    """
    返回已划分且已标准化的 (X_train, X_test, y_train, y_test, scaler_X, scaler_y)
    后续训练/推理可直接用。
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    df = df.drop(columns=['date'])

    X = df[['in-o2', '1-valve', 'in-nox']].values
    y = df[['out-nox']].values

    # 标准化: 把特征 X 和标签 y 分别“标准化”成均值为 0、标准差为 1 的分布
    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y)

    # 模型训练完后，如果你想把预测值还原回原始量纲（例如画图、算误差），可以：
    # y_pred_original = scaler_y.inverse_transform(y_pred)
    # 未来有新样本时，用同一套参数做变换：
    # X_new = scaler_X.transform(X_new)

    # 分割训练测试集: 训练集：占 80 %（因为 test_size=0.2）;测试集：占 20 %   
    # random_state=42 就是给随机数发生器上锁：数据集拆分、模型初始化、交叉验证等任何带随机性的步骤，
    # 只要 random_state 一样，每次跑代码得到的结果完全一致（行顺序、划分方式、随机初始化权重等都相同）。
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test, scaler_X, scaler_y


# --------------------------------------------------
# 4️⃣ 训练
# --------------------------------------------------
def train_model(model: torch.nn.Module,
                train_loader,
                criterion,
                optimizer,
                device,
                epochs: int = 100):
    """
    原地训练模型，每个 epoch 打印平均 loss。
    返回最后一轮的平均 loss。
    """
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)

            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}] | Loss: {avg_loss:.6f}")
    return avg_loss


# --------------------------------------------------
# 5️⃣ 测试
# --------------------------------------------------
@torch.no_grad()
def evaluate_model(model, test_loader, scaler_y, device):
    """
    在测试集上推理，返回反标准化后的预测值与真实值 (np.ndarray)
    preds_real, trues_real 形状均为 (N, 1)
    """
    model.eval()
    preds, trues = [], []

    for Xb, yb in test_loader:
        Xb = Xb.to(device)
        pred = model(Xb).cpu()
        preds.append(pred)
        trues.append(yb)

    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()

    preds_real = scaler_y.inverse_transform(preds)
    trues_real = scaler_y.inverse_transform(trues)

    print("✅ 测试集预测完成")
    return preds_real, trues_real


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

if __name__ == "__main__":
    csv_path = r"C:/Users/Clavi/Desktop/DRLlearning/SCRprocess/tuox_hd_20250710_0722.csv"
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = load_and_preprocess(csv_path)

    train_dataset = NOxDataset(X_train, y_train)
    test_dataset  = NOxDataset(X_test, y_test)
    
    train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader   = DataLoader(test_dataset,  batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = TransformerRegressor().to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_model(model, train_loader, criterion, optimizer, device, epochs=100)
    preds_real, trues_real = evaluate_model(model, test_loader, scaler_y, device)

