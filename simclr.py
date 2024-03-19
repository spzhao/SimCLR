import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):
        # cat之后的长度是 batch_size * n_views (比如 256*2 = 512)
        # 没有使用标注的label，是无监督的学习（或者自监督学习）
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        print('info_nce_loss label shape:', labels.shape)
        # unsqueeze 增加一个维度 ？
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        print('info_nce_loss label float shape:', labels.shape)
        labels = labels.to(self.args.device)

        # features 就是模型学习到的特征，shape = batch_size * out_dim
        print('info_nce_loss features before normalize shape:', features.shape, type(features))
        features = F.normalize(features, dim=1)

        print('info_nce_loss features normalize shape:', features.shape)

        similarity_matrix = torch.matmul(features, features.T)
        print('similarity_matrix shape', similarity_matrix.shape)  # 256 * 256
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        # 生成对角矩阵，长度为 labels.shape[0]，也就是 batch_size, batch_size * batch_size
        # mask 对角矩阵中，对角线位置值为 True，其他为值False
        # 矩阵中对角线值（需要画图才好理解）是自己与自己的变换进行对比，或者是自己与一种随机变化与自己的另一种随机变化进行对比的 “方差”
        # 所以对角线值是正样本对的特征对比，共有 （batch_size * n_views -1） 个
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        print('mask shape', mask.shape, labels.shape[0])
        # view 函数：改变tensor的形状
        # ~mask：将mask矩阵取反，True变成False，False变成True，~mask的矩阵中，对角线为False，其他为True
        # labels: 将mask矩阵取反后，维度就会少一个，重新view
        labels = labels[~mask].view(labels.shape[0], -1)
        print('labels after labels[~mask] shape', labels.shape)
        # 重新view重塑了similarity_matrix的矩阵，因为~mask对角线都为False，所以会少一列，similarity_matrix.shape = 256*255
        # 获取非对角线的向量值（不同数据向量之间的协方差？）
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        print('similarity_matrix shape', similarity_matrix.shape, similarity_matrix[:3])
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        # 正样本
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        print('positives shape', positives.shape, positives[:3])

        # select only the negatives the negatives
        # 负样本
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        # print('negatives shape', negatives.shape, negatives[:3])

        logits = torch.cat([positives, negatives], dim=1)
        # label 都置为0，这个是为什么呢？？？标志为0表示自监督学习？
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        # print('similarity_matrix shape', similarity_matrix.shape)
        # print('logits, labels:', logits, labels)
        return logits, labels

    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        print('train_loader len', len(train_loader))
        for epoch_counter in range(self.args.epochs):
            for images, _ in tqdm(train_loader):
                # print('images length', len(images)) # image: list[2]
                # print('images[1]', len(images[1]))  # 256，batch_size=256, 设置为多少就是一次加载多少个数据
                # print('images[0]', len(images[0]))  # 256

                images = torch.cat(images, dim=0)  # 二维转一维？从两个256的数组合并成为1个512长度的数组，每个数据是3x96x96（图像）的编码表示

                images = images.to(self.args.device)

                print('images after cat and to device len', len(images), images.shape) # 512, shape
                print('images after cat and to device 3', images[0].shape) # 3, 96, 96, 96是图片的size
                # print('images after cat and to device', images)

                with autocast(enabled=self.args.fp16_precision):
                    # features 是 batch_size个输入的特征，特征长度为128(通过out_dim设置)， shape = batch_size*128
                    features = self.model(images)
                    print('features.shape', features.shape)
                    logits, labels = self.info_nce_loss(features)
                    print('logits&labels.shape', logits.shape, labels.shape)
                    loss = self.criterion(logits, labels)
                    print('loss', loss.item())

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
