import torch
import numpy as np

#cutout�������Ϊ������ǿ��һ���֣�ͨ������ڵ�ͼ�������ʹģ���ܹ�ѧϰ��ȱʧ�����³���ԣ��Ӷ����ģ�͵ķ�������
class Cutout(object): #������ǿ����
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length): # ��ʼ��cutout���� n_holes:Ҫ��ͼ�����и��������������lengthÿ���и�����ı߳���������Ϊ��λ��
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img): # 1.�����˶��󱻵���ʱ����Ϊ�����ڴ���ͼ��
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        #2.����һ��tensor���͵�ͼ�����룬��ȡͼ��ĸ߶ȺͿ��
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32) #3.����һ��ȫ1��������󣬴�СΪͼ��ĸ߶�h�Ϳ��w
        #4.����ÿ��Ҫ�и������ �����ѡ���и�������������㣨x��y��
        #���ݸ����������С��������������Ͻǣ�x1,y1) �����½ǣ�x2,y2)������
        #���и����������ֵ��Ϊ0����ʾ��������ڵ�
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.
       #5.���������ת��Ϊtensor���ͣ��������С��չΪ������ͼ����ͬ
        #������ͼ���ж�Ӧ����λ�õ�����ֵ��Ϊ0 ��ʵ����������ڵ��������ڵ����ͼ��
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
