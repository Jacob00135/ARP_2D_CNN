import os
import cv2
import numpy as np
import pandas as pd
from time import time as get_timestamp
from collections import Counter
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from config import root_path, mri_3d_path, mri_2d_path


def split_dataset(data=None):
    # 读取数据集
    if data is None:
        data = pd.read_csv(os.path.join(root_path, 'datasets/data.csv'))

    # 打乱样本次序
    index = np.arange(data.shape[0], dtype='int')
    np.random.shuffle(index)
    data = data.iloc[index, :]

    """
    以下为分割数据集的代码
    在分割数据集时，要考虑：
    1. 测试集的类别为AD的样本，benefit尽可能地不为nan；其他的可以随意
    2. train_cn:valid_cn:test_cn ≈ train_ad:valid_ad:test_ad ≈ 6:2:2
    """

    # 处理test_ad
    y = data['AD'].values
    benefit = data['benefit'].values
    num_test_ad = int(np.ceil(sum(y == 1) * 0.2))
    not_benefit_ad = (y == 1) & (~np.isnan(benefit))
    test_set_boolean = np.zeros(y.shape[0], dtype='bool')

    if num_test_ad > sum(not_benefit_ad):
        index = np.where(not_benefit_ad)[0]
        test_set_boolean[index] = True

        r = num_test_ad - sum(not_benefit_ad)
        index = np.where((y == 1) & np.isnan(benefit))[0][:r]
        test_set_boolean[index] = True
    else:
        index = np.where(not_benefit_ad)[0][:num_test_ad]
        test_set_boolean[index] = True

    # 处理test_cn
    num_test_cn = int(np.ceil(sum(y == 0) * 0.2))
    index = np.where(y == 0)[0][:num_test_cn]
    test_set_boolean[index] = True

    # 分割出测试集
    test_set = data[test_set_boolean]
    data = data[~test_set_boolean]

    # 分割出训练集和验证集
    y = data['AD'].values
    valid_set_boolean = np.zeros(y.shape[0], dtype='bool')
    num_valid_ad = int(np.ceil(sum(y == 1) * 0.25))
    index = np.where(y == 1)[0][:num_valid_ad]
    valid_set_boolean[index] = True
    num_valid_cn = int(np.ceil(sum(y == 0) * 0.25))
    index = np.where(y == 0)[0][:num_valid_cn]
    valid_set_boolean[index] = True
    valid_set = data[valid_set_boolean]
    train_set = data[~valid_set_boolean]

    # 校验1
    y = test_set['AD'].values
    benefit = test_set['benefit'].values
    if sum((y == 1) & np.isnan(benefit)) != 0:
        y = train_set['AD'].values
        benefit = train_set['benefit'].values
        boolean = (y == 1) & (~np.isnan(benefit))
        assert sum(boolean) == 0, '校验1失败'
        y = valid_set['AD'].values
        benefit = valid_set['benefit'].values
        boolean = (y == 1) & (~np.isnan(benefit))
        assert sum(boolean) == 0, '校验1失败'

    # 校验2
    subset_list = [train_set, valid_set, test_set]
    ratio_list = [0.6, 0.2, 0.2]
    for category in [0, 1]:
        counter = []
        for subset in subset_list:
            counter.append(sum(subset['AD'].values == category))
        for count, ratio in zip(counter, ratio_list):
            assert abs(sum(counter) * ratio - count) < 2, '校验2失败'

    # 处理benefit
    test_set_benefit = test_set['benefit'].values
    test_set_benefit[np.isnan(test_set_benefit)] = 0
    test_set = test_set.drop(columns='benefit')
    test_set['benefit'] = test_set_benefit

    # 保存
    train_set.to_csv(os.path.join(root_path, 'datasets/train.csv'), index=False)
    valid_set.to_csv(os.path.join(root_path, 'datasets/valid.csv'), index=False)
    test_set.to_csv(os.path.join(root_path, 'datasets/test.csv'), index=False)


def get_dynamic_image(frames, normalized=True):
    """ Adapted from https://github.com/tcvrick/Python-Dynamic-Images-for-Action-Recognition"""
    """ Takes a list of frames and returns either a raw or normalized dynamic image."""
    
    def _get_channel_frames(iter_frames, num_channels):
        """ Takes a list of frames and returns a list of frame lists split by channel. """
        frames = [[] for channel in range(num_channels)]

        for frame in iter_frames:
            for channel_frames, channel in zip(frames, cv2.split(frame)):
                channel_frames.append(channel.reshape((*channel.shape[0:2], 1)))
        for i in range(len(frames)):
            frames[i] = np.array(frames[i])
        return frames


    def _compute_dynamic_image(frames):
        """ Adapted from https://github.com/hbilen/dynamic-image-nets """
        num_frames, h, w, depth = frames.shape

        # Compute the coefficients for the frames.
        coefficients = np.zeros(num_frames)
        for n in range(num_frames):
            cumulative_indices = np.array(range(n, num_frames)) + 1
            coefficients[n] = np.sum(((2*cumulative_indices) - num_frames) / cumulative_indices)

        # Multiply by the frames by the coefficients and sum the result.
        x1 = np.expand_dims(frames, axis=0)
        x2 = np.reshape(coefficients, (num_frames, 1, 1, 1))
        result = x1 * x2
        return np.sum(result[0], axis=0).squeeze()

    num_channels = frames[0].shape[2]
    #print(num_channels)
    channel_frames = _get_channel_frames(frames, num_channels)
    channel_dynamic_images = [_compute_dynamic_image(channel) for channel in channel_frames]

    dynamic_image = cv2.merge(tuple(channel_dynamic_images))
    if normalized:
        dynamic_image = cv2.normalize(dynamic_image, None, 0, 255, norm_type=cv2.NORM_MINMAX)
        dynamic_image = dynamic_image.astype('uint8')

    return dynamic_image


def transform_2d_mri(filenames):
    t = transforms.Resize((224, 224))
    start_time = get_timestamp()
    for fn in filenames:
        fp = os.path.join(mri_3d_path, fn)
        img = np.load(fp)
        img = np.expand_dims(img, -1)
        img = get_dynamic_image(img)
        img = Image.fromarray(img, 'L')
        img = t(img)
        img = np.array(img)
        img = np.expand_dims(img, 0)
        img = np.concatenate([img, img, img], 0)
        save_path = os.path.join(mri_2d_path, fn)
        np.save(save_path, img)
        print('{:.2f}s -- {}'.format(get_timestamp() - start_time, fn))


def generate_small_dataset(save_path, num_sample=600):
    # 检查参数是否合理
    if num_sample % 2 != 0:
        raise ValueError('num_sample必须是偶数！')
    data = pd.read_csv(os.path.join(root_path, 'datasets/data.csv'))
    y = data['AD'].values
    max_num_sample = min(Counter(y).values())
    if num_sample > max_num_sample * 2:
        raise ValueError('num_sample太大，需要小于等于: {}'.format(max_num_sample * 2))

    # 生成小样本
    num_cn = num_sample // 2
    num_ad = num_cn
    index0 = np.where(y == 0)[0][:num_cn]
    index1 = np.where(y == 1)[0][:num_ad]
    index = np.append(index0, index1)
    small_data = data.iloc[index, :]
    small_data.to_csv(save_path, index=False)


def npy_to_image():
    save_dir_path = os.path.join(root_path, 'datasets/2d_image')
    if not os.path.exists(save_dir_path):
        os.mkdir(save_dir_path)
    for fn in tqdm(os.listdir(mri_2d_path)):
        start_time = get_timestamp()

        fp = os.path.join(mri_2d_path, fn)
        image_array = np.load(fp)
        rows, cols = image_array.shape[1:]
        new_image_array = np.zeros((rows, cols, 3), dtype=image_array.dtype)
        for r in range(rows):
            for c in range(cols):
                for z in range(3):
                    new_image_array[r, c, z] = image_array[z, r, c]
        image = Image.fromarray(new_image_array)
        save_path = os.path.join(save_dir_path, '{}.png'.format(fn.rsplit('.', 1)[0]))
        image.save(save_path)


class DataAugmentation(object):

    def __init__(self, data):
        self.data = data

        self.aug_function = [
            'shrink', 'magnify', 'clockwise_rotation', 'anticlockwise_rotation',
            'translate_up', 'translate_down', 'translate_left', 'translate_right'
        ]

    def generate(self):
        """
        数据分布：{0.0: 4699, 1.0: 1507}
        可选的变换方式：
        1. 旋转：顺时针、逆时针
        2. 平移：上、下、左、右
        3. 缩放：缩小、放大
        每一种变换方式，需要生成：(4699 - 1507) / 8 = 399个样本
        """

        # TODO 添加进度条，或者输出处理成功的信息和时间，以便查看生成进度

        # 将DataFrame转换为List[Dict]，方便后续添加样本的操作
        records = self.data.to_dict(orient='records')

        # 对每一个样本都添加dir_path、is_aug_data列
        for record in records:
            record['dir_path'] = mri_2d_path
            record['is_aug_data'] = 0

        # 计算两个类别的样本量、要生成的样本量
        counter = Counter([r['AD'] for r in records])
        num_cn = counter[0]
        num_ad = counter[1]
        num_generate = int((num_cn - num_ad) / len(self.aug_function))

        # 创建目录，保存增强后的图片
        aug_data_dir = []
        for func_name in self.aug_function:
            path = os.path.join(root_path, 'datasets/aug_2d_mri', func_name)
            if not os.path.exists(path):
                os.mkdir(path)
            aug_data_dir.append(path)

        # 分离AD类的样本
        ad_records = []
        for record in records:
            if record['AD'] == 1:
                ad_records.append(record)

        # 生成样本
        new_ad_records = []
        for i, func_name in enumerate(self.aug_function):
            progress_bar = tqdm(range(num_generate), desc='{}'.format(func_name))
            for j in progress_bar:
                k = (i * num_generate + j) % num_ad
                filename = ad_records[k]['filename']

                getattr(self, func_name)(
                    src_path=os.path.join(mri_2d_path, filename),
                    dst_path=os.path.join(aug_data_dir[i], filename)
                )

                new_record = ad_records[k].copy()
                new_record['dir_path'] = aug_data_dir[i]
                new_record['is_aug_data'] = 1
                new_ad_records.append(new_record)

        # 添加生成的样本的信息到总数据里，并保存成csv
        records.extend(new_ad_records)
        records_df = pd.DataFrame(records)
        save_path = os.path.join(root_path, 'datasets/aug_data.csv')
        records_df.to_csv(save_path, index=False)

    @staticmethod
    def shrink(src_path, dst_path):
        # 载入原图片，并进行缩小
        src_img = np.load(src_path).transpose((1, 2, 0))
        src_size = src_img.shape[0]
        dst_size = int(src_size * 0.9)
        shrink_img = cv2.resize(
            src=src_img,
            dsize=(dst_size, dst_size),
            interpolation=cv2.INTER_AREA
        )

        # 缩小后，图片形状比原图更小，所以需要填充黑边，以达到跟原图同样的大小
        up_fill = int((src_size - dst_size) / 2)
        bottom_fill = dst_size + up_fill
        left_fill = up_fill
        right_fill = bottom_fill
        dst_img = np.zeros(src_img.shape, dtype=src_img.dtype)
        dst_img[up_fill:bottom_fill, left_fill:right_fill] = shrink_img
        fill_value = 0
        for r in range(up_fill):
            dst_img[r, :] = fill_value
        for r in range(bottom_fill, src_size):
            dst_img[r, :] = fill_value
        for c in range(left_fill):
            dst_img[:, c] = fill_value
        for c in range(right_fill, src_size):
            dst_img[:, c] = fill_value

        # 保存图片为npy
        np.save(dst_path, dst_img)

        return dst_img

    @staticmethod
    def magnify(src_path, dst_path):
        # 载入原图片，并进行放大,
        src_img = np.load(src_path).transpose((1, 2, 0))
        src_size = src_img.shape[0]
        dst_size = int(src_size * 1.1)
        magnify_img = cv2.resize(
            src=src_img,
            dsize=(dst_size, dst_size),
            interpolation=cv2.INTER_CUBIC
        )

        # 放大后，图片形状比原图更大，所以需要裁剪边界，以达到跟原图同样的大小
        up_clip = int((dst_size - src_size) / 2)
        bottom_clip = src_size + up_clip
        left_clip = up_clip
        right_clip = bottom_clip
        dst_img = magnify_img[up_clip:bottom_clip, left_clip:right_clip]

        # 保存图片为npy
        np.save(dst_path, dst_img)

        return dst_img

    @staticmethod
    def clockwise_rotation(src_path, dst_path):
        src_img = np.load(src_path).transpose((1, 2, 0))
        rows, cols = src_img.shape[:2]
        angle = -30
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        dst_img = cv2.warpAffine(src_img, rotation_matrix, (cols, rows))

        np.save(dst_path, dst_img)

        return dst_img

    @staticmethod
    def anticlockwise_rotation(src_path, dst_path):
        src_img = np.load(src_path).transpose((1, 2, 0))
        rows, cols = src_img.shape[:2]
        angle = 30
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        dst_img = cv2.warpAffine(src_img, rotation_matrix, (cols, rows))

        np.save(dst_path, dst_img)

        return dst_img

    @staticmethod
    def translate_up(src_path, dst_path):
        src_img = np.load(src_path).transpose((1, 2, 0))
        dst_img = np.zeros(src_img.shape, dtype=src_img.dtype)
        height = src_img.shape[0]
        translate_distance = int(height * 0.1)
        dst_img[:height - translate_distance, :] = src_img[translate_distance:, :]
        np.save(dst_path, dst_img)

        return dst_img

    @staticmethod
    def translate_down(src_path, dst_path):
        src_img = np.load(src_path).transpose((1, 2, 0))
        dst_img = np.zeros(src_img.shape, dtype=src_img.dtype)
        height = src_img.shape[0]
        translate_distance = int(height * 0.1)
        dst_img[translate_distance:, :] = src_img[:height - translate_distance, :]
        np.save(dst_path, dst_img)

        return dst_img

    @staticmethod
    def translate_left(src_path, dst_path):
        src_img = np.load(src_path).transpose((1, 2, 0))
        dst_img = np.zeros(src_img.shape, dtype=src_img.dtype)
        width = src_img.shape[1]
        translate_distance = int(width * 0.1)
        dst_img[:, :width - translate_distance] = src_img[:, translate_distance:]
        np.save(dst_path, dst_img)

        return dst_img

    @staticmethod
    def translate_right(src_path, dst_path):
        src_img = np.load(src_path).transpose((1, 2, 0))
        dst_img = np.zeros(src_img.shape, dtype=src_img.dtype)
        width = src_img.shape[1]
        translate_distance = int(width * 0.1)
        dst_img[:, translate_distance:] = src_img[:, :width - translate_distance]
        np.save(dst_path, dst_img)

        return dst_img


if __name__ == '__main__':
    """数据增强
    data = pd.read_csv(os.path.join(root_path, 'datasets/data.csv'))
    da = DataAugmentation(data)
    da.generate()
    """
    data = pd.read_csv(os.path.join(root_path, 'datasets/aug_data.csv'))
    split_dataset(data)
    # transform_2d_mri()
    # npy_to_image()
    """
    generate_small_dataset(
        save_path='./datasets/data3014.csv',
        num_sample=3014
    )
    split_dataset(pd.read_csv('./datasets/data3014.csv'))
    """

