import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import cv2 as cv
import os, shutil
from pathlib import Path
from PIL import Image

# 获得该文件夹下所有jpg图片路径
p = Path(r"C:\Users\yuyang\Desktop\image")
files = list(p.glob("**/*.png"))

# 图像预处理，定义了一系列图像预处理操作，包括图像大小调整、转换为张量（tensor）和标准化。
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

images = []
paths = []

for file in files:
    img = Image.open(file)
    img = transform(img)
    images.append(img)
    paths.append(file)


# 加载预训练的ResNet-50模型
model = models.resnet50(pretrained=True)
model.eval() # 设置为评估模式，以确保不进行梯度计算。


# 对图像进行预测
tensor_images = []
tensor_images = torch.stack(images)

# 使用ResNet-50模型进行预测
with torch.no_grad():# 上下文管理器，我们禁用梯度计算，以减少内存消耗
    predictions = model(tensor_images)
pred_images = predictions.reshape(tensor_images.shape[0], -1)


# k = 3  # 2个类别
# # 层次聚类
# hmodel = AgglomerativeClustering(n_clusters=k)
# hmodel.fit(pred_images)
# hpredictions = hmodel.labels_
# print(hpredictions)  # 预测的类别

# k = 2  # 聚类数目
#
# # K-means聚类
# kmodel = KMeans(n_clusters=k, random_state=8)
# hpredictions = kmodel.fit_predict(pred_images)
#
# print(hpredictions)  # 预测的类别

dbscan = DBSCAN(eps=0.5, min_samples=5)
cluster_labels = dbscan.fit_predict(pred_images)
num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
print("自动确定的类别数目：", num_clusters)

#0：dog 1：cat
'''
sil = []
kl = []
kmax = 10
for k in range(2, kmax + 1):
    hclustering = AgglomerativeClustering(n_clusters=k)
    labels = hclustering.fit_predict(pred_images)
    sil.append(silhouette_score(pred_images, labels, metric='euclidean'))
    kl.append(k)
'''
# for i in ["0", "1"]:
#     os.mkdir(r"C:\Users\yuyang\Desktop\cluster_" + str(i))
#
#  # 复制文件，保留元数据 shutil.copy2('来源文件', '目标地址')
# for i in range(len(paths)):
#     if hpredictions[i] == 0:
#         shutil.copy2(paths[i], r"C:\Users\yuyang\Desktop\cluster_0")
#     elif hpredictions[i] == 1:
#         shutil.copy2(paths[i], r"C:\Users\yuyang\Desktop\cluster_1")

'''
plt.plot(kl, sil,'r-.p')
plt.ylabel('Silhoutte Score')
plt.xlabel('K')
plt.show()
'''
