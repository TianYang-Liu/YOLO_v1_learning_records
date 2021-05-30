import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class VOCDataset(Dataset):
    def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.S = S
        self.B = B
        self.C = C
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # 打开csv的地index行第一列
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        # 将所有label存在二维列表
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]
            boxes.append([class_label, x, y, width, height])

        # 打开图片
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)

        # 将box转换为Tensor以便transformer中使用
        boxes = torch.Tensor(boxes)

        # transformer
        if self.transform:
            image,box = self.transform(image,boxes)

        # 制作label矩阵
        label_matrix = torch.zeros((self.S, self.S, self.B * 5 + self.C))  # 虽然初始化了30个，但是后面5个是没用的
        for box in boxes:
            # 将在transformer中转换成tensor的box再转换为list
            class_label, x, y, width, height = box.tolist()

            # 确保class_label是整数
            class_label = int(class_label)

            ##将x, y变为相对于grid的归一化，原始x，y是相对与原图的归一化
            # 找到x,y所属的grid的顶点坐标
            i, j = int(y * self.S), int(x * self.S)

            # x,y所属的cell的顶点坐标,i,j分别代表cell row and cell column

            x_cell, y_cell = x * self.S - j, y * self.S - i
            # 矩阵是按行索引, 也就是对应y值
            # x,y相对于cell的坐标

            # 对宽和高不处理，因为原始数据宽和高就是相对于原图归一化，以下是相对于cell的尺寸
            # width_cell,height_cell=(width*self.S,height*self.S)

            if label_matrix[i, j, 20] == 0:
                # 将confidence的score设为1
                label_matrix[i, j, 20] = 1
                box_coordinates = torch.tensor([x_cell, y_cell, width, height])

                # 写入坐标
                label_matrix[i, j, 21:25] = box_coordinates
                label_matrix[i, j, class_label] = 1

            return image, label_matrix
