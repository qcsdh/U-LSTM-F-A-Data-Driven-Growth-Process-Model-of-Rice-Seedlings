import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# 定义数据集类
class RiceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(root_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, img_name

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载数据集
dataset = RiceDataset(r'C:\Users\10691\Desktop\leaf-all\2o-p', transform=transform)

# 定义数据加载器
data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# 加载已训练好的模型
model = RiceCNN()
model.load_state_dict(torch.load(r'C:\Users\10691\Desktop\leaf-all\model\model.pt'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# 预测并保存结果
results = {}  # 用于保存图像名和叶龄的对应关系

with torch.no_grad():
    for images, image_names in data_loader:
        images = images.to(device)

        outputs = model(images)
        _, predicted = outputs.max(1)

        # 逐个处理每个样本的预测结果
        for i in range(len(images)):
            image_name = image_names[i]  # 获取图像文件名
            pred = predicted[i].item()  # 预测的叶龄

            # 输出图像名和叶龄
            print('Image name:', image_name)
            print('Predicted leaf age:', pred)
            print('---------------------------------')

            # 保存图像名和叶龄的对应关系
            results[image_name] = pred

# 保存图像名和叶龄对应关系到文件
with open(r'C:\Users\10691\Desktop\leaf-all\predictions\predictions.txt', 'w') as file:
    for image_name, pred in results.items():
        file.write(f'Image name: {image_name}, Predicted leaf age: {pred}\n')