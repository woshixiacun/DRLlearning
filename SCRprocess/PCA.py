###########################手工版（NumPy 一步一步算协方差、特征分解）######################################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# 1. 数据准备
X, y = load_iris(return_X_y=True)
X = X - X.mean(axis=0)                 # 去均值（等同于 StandardScaler 的零均值）

# 2. 协方差矩阵 + 特征分解
C = np.cov(X, rowvar=False)            # 4×4 协方差
eig_vals, eig_vecs = np.linalg.eigh(C) # 特征值、特征向量

# 3. 排序并取前 2 个主成分
idx = np.argsort(eig_vals)[::-1]
W = eig_vecs[:, idx[:2]]               # 4×2 投影矩阵

Z = X @ W                              # 150×2 降维结果

print('解释方差比:', (eig_vals[idx[:2]] / eig_vals.sum()).round(3))
plt.scatter(Z[:, 0], Z[:, 1], c=y, cmap='tab10')
plt.title('PCA on Iris (NumPy manual)')
plt.xlabel('PC1'); plt.ylabel('PC2')
plt.show()



###########################一行代码版（scikit-learn 高层 API）######################################

# # pip install scikit-learn matplotlib
# from sklearn.datasets import load_iris
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt

# X, y = load_iris(return_X_y=True)      # 150×4 数据
# X = StandardScaler().fit_transform(X)  # 必须标准化！

# pca = PCA(n_components=2)              # 降到 2 维
# Z = pca.fit_transform(X)               # 150×2

# print('解释方差比:', pca.explained_variance_ratio_)
# plt.scatter(Z[:, 0], Z[:, 1], c=y, cmap='tab10')
# plt.title('PCA on Iris (sklearn)')
# plt.xlabel('PC1'); plt.ylabel('PC2')
# plt.show()



