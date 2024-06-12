import os
import torch
import numpy as np
from tqdm import tqdm
from config import root_path
from dataloader import MRI2D
from models import CNN


def build_model(device):
    net = CNN()
    if device != 'cpu':
        net = net.to(device)

    return net


def predict(net, dataset, device, desc=None):
    net.eval()
    prediction = np.zeros((len(dataset), 2), dtype='float32')
    if device == 'cpu':
        with torch.no_grad():
            progress_bar = tqdm(dataset, desc=desc)
            j = 0
            for image, label in progress_bar:
                image = torch.tensor(np.expand_dims(image, 0))
                output = net(image)
                prediction[j, :] = output.numpy()
                j = j + 1
    else:
        with torch.no_grad():
            progress_bar = tqdm(dataset, desc=desc)
            j = 0
            for image, label in progress_bar:
                image = torch.tensor(np.expand_dims(image, 0)).to(device)
                output = net(image)
                prediction[j, :] = output.cpu().squeeze().numpy()
                j = j + 1

    return prediction


def main(model_name, device):
    # 找到所有模型
    checkpoint_dir = os.path.join(root_path, 'checkpoints', model_name)
    filenames = sorted(
        os.listdir(checkpoint_dir),
        key=lambda fn: int(fn.rsplit('.', 1)[0].rsplit('_', 1)[1])
    )

    # 预测
    test_set = MRI2D(os.path.join(root_path, 'datasets/test.csv'))
    net = build_model(device)
    prediction = np.zeros((len(filenames), len(test_set), 2), dtype='float32')
    for i, filename in enumerate(filenames):
        model_path = os.path.join(checkpoint_dir, filename)
        net.load_state_dict(torch.load(model_path, map_location=device))
        prediction[i, :, :] = predict(net, test_set, device, filename.rsplit('.', 1)[0])

    # 保存预测结果
    save_dir_path = os.path.join(root_path, 'eval_result', model_name)
    if not os.path.exists(save_dir_path):
        os.mkdir(save_dir_path)
    save_path = os.path.join(save_dir_path, 'prediction.npy')
    np.save(save_path, prediction)


if __name__ == '__main__':
    main(
        model_name='xjy_20240611',
        device='cpu'
    )
