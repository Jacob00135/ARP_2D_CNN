[TOC]

复现论文：《Dynamic Image for 3D MRI Image Alzheimer’s Disease Classification》

# 1. 数据预处理

数据预处理与config.py和data_preprocess.py相关。

首先修改config.py中的变量`mri_3d_path`为存放3D影像的目录。

## 1.1 将3D影像转换为2D图片

在datasets目录中，创建2d_mri子目录，然后在data_preprocess.py中，调用函数`transform_2d_mri`来将3D影像转换成2D图片：

```python
filenames = pd.read_csv(os.path.join(root_path, 'datasets/data.csv'))['filename'].values
transform_2d_mri(filenames)
```

然后执行：

```bash
python data_preprocess.py
```

代码运行时，将把转换后的2D图片以npy格式保存在`datasets/2d_mri`

## 1.2 （可选）数据增强

在datasets目录中，创建aug_2d_mri子目录，然后在data_preprocess.py中编写以下代码进行数据增强：

```python
data = pd.read_csv(os.path.join(root_path, 'datasets/data.csv'))
da = DataAugmentation(data)
da.generate()
```

然后执行：

```bash
python data_preprocess.py
```

代码运行时将对2D图片进行数据增强，把增强后的数据保存在`datasets/aug_2d_mri`中，同时在datasets目录下生成`aug_data.csv`，这个csv文件同时包含了原图片和增强后图片的信息

## 1.3 分割数据集

如果进行了1.2的数据增强，在data_preprocess.py中编写以下代码进行数据预处理：

```python
data = pd.read_csv(os.path.join(root_path, 'datasets/aug_data.csv'))
split_dataset(data)
```

如果没有进行1.2的数据增强，那么代码应该是：

```python
data = pd.read_csv(os.path.join(root_path, 'datasets/data.csv'))
split_dataset(data)
```

然后执行：

```bash
python data_preprocess.py
```

运行完毕后，将把数据集分为训练集、验证集、测试集，保存在`datasets/train.csv`、`datasets/valid.csv`和`datasets/test.csv`

# 2. 训练模型

在训练之前，需要在项目根目录下创建目录`checkpoints`，用于在训练时保存模型文件。

训练模型的代码是train.py，在执行此代码之前，需要指定一些参数，然后调用train方法，如下示例：

```python
mt = ModelTrainer(
    model_name='your_model_name',  # 模型名称，影响模型文件的保存路径、模型验证结果的保存路径
    device='cuda:1',  # 使用的GPU编号
    batch_size=16,  # 训练时载入数据的批次大小
    init_lr=0.00001,  # 初始学习率
    weight_decay=0.01  # 权重衰减参数
)
mt.train(100)  # 传入训练的轮次（epoch）
```

然后执行：

```bash
python train.py
```

在训练过程中，将会以进度条的形式显示每一轮的训练进度，并在每一轮训练结束后，计算训练集和验证集的Accuracy输出到控制台，并且保存本轮的模型到：`checkpoints/<model_name>`

# 3. 模型预测

首先在项目根目录下创建目录`eval_result`，用于保存预测结果。

模型预测代码是`model_predict.py`，使用`checkpoints/<model_name>`中保存的所有模型对测试集进行预测，并把结果保存为`eval_result/<model_name>/prediction.npy`，这是一个float32的数组，形状为(模型文件数量, 测试集样本量, 2)。

在`model_predict.py`中调用`main`函数，指定两个参数：

```python
main(
    model_name='your_model_name',  # 训练好的模型的名称
    device='cpu'   # 使用的显卡编号，若不使用显卡，设为"cpu"
)
```

然后执行：

```bash
python model_predict.py
```

# 4. 计算指标

本项目中计算的指标为`Accuracy`、`Precision`、`Recall`、`F1-Score`、`AUC`、`AP`，计算指标的代码是`compute_performance.py`，调用其main函数，需要更改参数：

```python
main(model_name='your_model_name')  # 训练好的模型的名称
```

然后执行：

```bash
python compute_performance.py
```

代码执行时，将会读取`eval_result/<model_name>/prediction.npy`和`datasets/test.csv`来进行指标的计算，并将计算结果保存为csv，路径为`eval_result/<model_name>/performance.csv`

# 5. 绘图

绘图代码为`draw_graph.py`，在执行之前，需要更改模型的名称：

```python
model_name = 'your_model_name'
```


然后执行：

```bash
python draw_graph.py
```

运行完毕后，将把绘制的曲线图保存在：`eval_result/<model_name>`