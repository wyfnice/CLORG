import torch
import numpy as np

#cutout类可以作为数据增强的一部分，通过随机遮挡图像的区域，使模型能够学习对缺失区域的鲁棒性，从而提高模型的泛化能力
class Cutout(object): #数据增强技术
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length): # 初始化cutout对象 n_holes:要从图像中切割出得区域数量，length每个切割区域的边长（以像素为单位）
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img): # 1.定义了对象被调用时的行为，用于处理图像
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        #2.接受一个tensor类型的图像输入，获取图像的高度和宽度
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32) #3.创建一个全1的掩码矩阵，大小为图像的高度h和宽度w
        #4.对于每个要切割的区域 ：随机选择切割区域中心坐标点（x，y）
        #根据给定的区域大小，计算区域的左上角（x1,y1) 和右下角（x2,y2)的坐标
        #将切割区域的掩码值设为0，表示这个区域被遮挡
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.
       #5.将掩码矩阵转换为tensor类型，并将其大小扩展为与输入图像相同
        #将输入图像中对应掩码位置的像素值置为0 ，实现了区域的遮挡，返回遮挡后的图像
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
