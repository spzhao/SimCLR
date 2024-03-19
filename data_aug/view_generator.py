import numpy as np

np.random.seed(0)


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        # print('ContrastiveLearningViewGenerator __call__', type(x))
        # base_transform 是一个函数，n_views = 2 是一次要比较的对象是2个: z_i, z_j
        # 进行了两次 transform(x), 但是两次进行的过程都是随机（xxRandom()）的，所以出来的图片矩阵（embeding）是类似的
        res = [self.base_transform(x) for i in range(self.n_views)]
        # print(len(res[0][0]), res[0][0].shape)
        # print(res[0][0], res[1][0])
        # print(res[0][-1], res[1][-1])
        return res
