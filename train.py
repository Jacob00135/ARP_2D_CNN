import os
import torch
import numpy as np
from tqdm import tqdm
from time import time as get_timestamp
from torch.utils.data import DataLoader
from config import root_path, mri_2d_path
from dataloader import MRI2D
from models import CNN


class ModelTrainer(object):

    def __init__(
        self,
        model_name, device, batch_size=16, init_lr=0.0001,
        weight_decay=0.001
    ):
        # 初始化变量
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.weight_decay = weight_decay

        self.num_classes = 2
        self.epoch = 0
        self.steps = 0
        self.checkpoint_path = os.path.join(root_path, 'checkpoints', model_name)
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)

        # 初始化数据载入器
        train_set = MRI2D(os.path.join(root_path, 'datasets/train.csv'))
        valid_set = MRI2D(os.path.join(root_path, 'datasets/valid.csv'))
        self.train_set = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, drop_last=False)
        self.valid_set = DataLoader(valid_set, batch_size=self.batch_size, shuffle=True, drop_last=False)
        print('成功载入数据')

        # 载入模型
        self.net = CNN().to(self.device)
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.init_lr,
            weight_decay=self.weight_decay
        )
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)
        self.loss = torch.nn.CrossEntropyLoss()
        print('成功载入模型')

    def train_an_epoch(self):
        self.net.train()
        batch_loss_list = []
        progress_bar = tqdm(self.train_set, desc='Epoch {}'.format(self.epoch))
        for batch_image, batch_label in progress_bar:
            batch_image = batch_image.to(self.device)
            batch_label = batch_label.long().to(self.device)

            self.optimizer.zero_grad()
            batch_output = self.net(batch_image)
            batch_loss = self.loss(batch_output, batch_label)

            batch_loss.backward()
            self.optimizer.step()

            batch_loss_list.append(batch_loss.item())

            # 保存模型
            self.steps = self.steps + 1
            if self.need_to_save_model():
                filename = 'steps_{}.pth'.format(self.steps)
                self.save_model(filename)

            # break  # TODO

        loss_mean = sum(batch_loss_list) / len(batch_loss_list)

        return loss_mean
 
    def train(self, num_epoch):
        for epoch in range(1, num_epoch + 1):
            epoch_start_time = get_timestamp()
            self.epoch = epoch

            self.train_an_epoch()

            # 保存模型
            """
            filename = 'epoch_{}.pth'.format(epoch)
            self.save_model(filename)
            """

            # 计算准确率和loss
            train_loss, train_accuracy = self.eval_model('train')
            valid_loss, valid_accuracy = self.eval_model('valid')

            # 调整学习率
            self.lr_scheduler.step()

            # 输出日志
            print('{:.0f}s - loss = {:.8f} - val_loss = {:.8f} - acc = {:.8f} - val_acc = {:.8f}'.format(
                get_timestamp() - epoch_start_time, train_loss, valid_loss, train_accuracy, valid_accuracy
            ))
            print()

            # break  # TODO

    def need_to_save_model(self):
        if self.epoch <= 1:
            return True
        if self.epoch <= 5 and self.steps % 5 == 0:
            return True
        if self.epoch <= 10 and self.steps % 50 == 0:
            return True
        if self.epoch <= 50 and self.steps % 100 == 0:
            return True
        return self.steps % 200 == 0

    def save_model(self, filename):
        save_path = os.path.join(self.checkpoint_path, filename)
        torch.save(self.net.state_dict(), save_path)

    def load_model(self, filepath):
        self.net.load_state_dict(torch.load(filepath, map_location=self.device))

    def predict(self, stage):
        # 载入数据
        dataset = MRI2D(os.path.join(root_path, 'datasets/{}.csv'.format(stage)))
        dataset = DataLoader(dataset, batch_size=1, shuffle=False)
        num_dataset = len(dataset)

        # 初始化变量
        prediction = np.zeros((num_dataset, self.num_classes), dtype='float32')
        labels = np.zeros(num_dataset, dtype='int32')
        loss_list = []

        # 预测
        self.net.eval()
        with torch.no_grad():
            progress_bar = tqdm(dataset, desc='Predicting...')
            i = 0
            for image, label in progress_bar:
                image = image.to(self.device)
                label = label.long().to(self.device)

                output = self.net(image)
                loss = self.loss(output, label)

                prediction[i, :] = output.cpu().squeeze().numpy()
                labels[i] = label.cpu().squeeze().numpy()
                loss_list.append(loss.item())

                i = i + 1

                # break  # TODO

        loss_mean = sum(loss_list) / len(loss_list)

        return prediction, labels, loss_mean

    def eval_model(self, stage):
        prediction, labels, loss_mean = self.predict(stage)
        pred_labels = prediction.argmax(axis=1)
        accuracy = sum(pred_labels == labels) / labels.shape[0]

        return loss_mean, accuracy

    def estimate_save_model_size(self, num_epoch):
        # 计算总大小
        model_size = 450657517  # 保存的单个模型的大小，单位为B
        num_model = 0
        num_batch = len(self.train_set)
        for epoch in range(1, num_epoch + 1):
            self.epoch = epoch
            for _ in range(num_batch):
                self.steps = self.steps + 1
                num_model = num_model + int(self.need_to_save_model())
        total_size = num_model * model_size

        # 换算单位
        human_total_size = total_size
        for u in ['B', 'KB', 'MB', 'GB', 'TB']:
            if human_total_size >= 1024:
                human_total_size = human_total_size / 1024
            else:
                human_total_size = '{} {}'.format(human_total_size, u)
                break
        print('预估会保存的模型个数：{}'.format(num_model))
        print('保存的模型的总大小：{} ({} B)'.format(human_total_size, total_size))

        self.epoch = 0
        self.steps = 0


if __name__ == '__main__':
    mt = ModelTrainer(
        model_name='xjy_20240613',
        device='cuda:1',
        batch_size=16,
        init_lr=0.00001,
        weight_decay=0.01
    )
    mt.estimate_save_model_size(100)
    mt.train(100)
    """
    mt.load_model(os.path.join(mt.checkpoint_path, 'epoch_100.pth'))
    test_loss, test_accuracy = mt.eval_model('test')
    print('test_loss = {:.8f} - test_accuracy = {:.8f}'.format(
        test_loss, test_accuracy
    ))
    """
