import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score,
    recall_score, average_precision_score
)
from config import root_path


def one_hot(array, num_classes):
    result = np.zeros((array.shape[0], num_classes), dtype=array.dtype)
    for i in range(array.shape[0]):
        result[i, int(array[i])] = 1

    return result


def main(model_name):
    # 载入计算指标所需的变量
    data = pd.read_csv(os.path.join(root_path, 'datasets/test.csv'))
    prediction_path = os.path.join(root_path, 'eval_result/{}/prediction.npy'.format(model_name))
    prediction = np.load(prediction_path)  # (模型数量, 测试集样本量, 2)
    y_true = data['AD'].values
    y_true_one_hot = one_hot(y_true, 2)
    benefit = data['benefit'].values
    benefit_sum = sum(benefit)
    filenames = sorted(
        os.listdir(os.path.join(root_path, 'checkpoints', model_name)),
        key=lambda fn: int(fn.rsplit('.', 1)[0].rsplit('_', 1)[1])
    )

    # 计算指标
    result = {
        'model_name': filenames,
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': [],
        'ap': [],
        'benefit': []
    }
    for i in tqdm(range(prediction.shape[0]), desc='Computing...'):
        y_score = prediction[i]
        y_pred = y_score.argmax(axis=1)
        result['accuracy'].append(accuracy_score(y_true, y_pred))
        result['precision'].append(precision_score(y_true, y_pred, zero_division=np.nan))
        result['recall'].append(recall_score(y_true, y_pred))
        result['f1'].append(f1_score(y_true, y_pred))
        result['auc'].append(roc_auc_score(y_true_one_hot, y_score))
        result['ap'].append(average_precision_score(y_true_one_hot, y_score))
        result['benefit'].append(benefit[y_true == y_pred].sum() / benefit_sum)

    df = pd.DataFrame(result)
    print(df)
    df.to_csv(os.path.join(root_path, 'eval_result/{}/performance.csv'.format(model_name)), index=False)


if __name__ == '__main__':
    main('xjy_20240611')
