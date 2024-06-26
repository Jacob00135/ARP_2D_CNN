import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import root_path


def draw_full_plot(x, y, xlabel, save_path):
    fig = plt.figure(figsize=(4, 4), dpi=1200)
    ax = plt.subplot(1, 1, 1)
    plt.scatter(
        x,
        y,
        marker='o',
        c='#6868ff',
        s=15,
        lw=0.2,
        ec='#555555',
        zorder=2
    )
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(np.arange(0.0, 1.1, 0.1))
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.xlabel(xlabel)
    plt.ylabel('benefit')
    plt.grid(True, c='#eeeeee', ls='--', lw=0.5, zorder=0)
    ax.set_aspect('equal', adjustable='box')
    # fig.set_size_inches(3.5, 3.5)
    # plt.subplots_adjust(left=0.15, bottom=0.13, top=0.93, right=0.95)
    plt.savefig(save_path)
    plt.close()


def filtrate_dense_point(x, y, x_threshold=0.01, y_threshold=0.01, include_best_point=False):
    if x.shape[0] != y.shape[0]:
        raise ValueError('x和y长度不一致！')

    length = x.shape[0]
    filtrate_boolean = np.ones(length, dtype='bool')
    for i in range(length - 2):
        for j in range(i + 1, length):
            if abs(x[j] - x[i]) <= x_threshold and abs(y[j] - y[i]) <= y_threshold:
                filtrate_boolean[j] = False

    if not include_best_point:
        filtrate_boolean[x.argmax()] = True
        filtrate_boolean[y.argmax()] = True

    x = x[filtrate_boolean]
    y = y[filtrate_boolean]

    return x, y


if __name__ == '__main__':
    model_name = 'xjy_20240613'
    data = pd.read_csv(os.path.join(root_path, 'eval_result/{}/performance.csv'.format(model_name)))
    y = data['benefit'].values
    for y_name in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'ap']:
        x = data[y_name].values
        filtered_x, filtered_y = filtrate_dense_point(x, y)
        save_path = os.path.join(root_path, 'eval_result/{}/{}.png'.format(model_name, y_name))
        draw_full_plot(filtered_x, filtered_y, y_name, save_path)
