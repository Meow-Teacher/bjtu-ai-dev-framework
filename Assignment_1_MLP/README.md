# MLP 神经网络实现
基于 Python 实现的多层感知机（MLP）神经网络项目，提供了完整的训练和推理功能，包含了一个基于 Streamlit 的 Web 界面。

## 项目结构

```
Assignment_1_MLP/
├── app.py          # Streamlit Web 应用界面
├── api.py          # FastAPI 后端服务
├── mlp.py          # MLP 神经网络核心实现
├── boston.csv      # 示例数据集
└── models/         # 模型存储目录
```

## 功能特性

### 1. 神经网络核心功能
- 支持多种激活函数：Sigmoid、Tanh、ReLU、Leaky ReLU、Softmax、Linear
- 支持多种优化器：SGD、Momentum、RMSProp、Adam
- 支持多种正则化方法：L1、L2、Elastic Net
- 支持多种权重初始化方法：Zero、Random、Xavier、He
- 支持批量归一化（Batch Normalization）
- 支持 Dropout 正则化
- 支持并行训练
- 支持早停（Early Stopping）

### 2. 模型类型
- 分类器（MLPClassifier）：用于分类任务
- 回归器（MLPRegressor）：用于回归任务

### 3. Web 界面功能
- 模型创建和配置
- 数据上传和预处理
- 模型训练和监控
- 模型评估和可视化
- 预测功能

### 4. API 服务
- RESTful API 接口
- 异步训练支持
- 模型持久化存储
- 训练状态监控

## 使用方法

### 1. 启动 API 服务
```bash
uvicorn api:app --reload
```

### 2. 启动 Web 界面
```bash
streamlit run app.py
```

### 3. 使用 Web 界面
1. 打开浏览器访问 `http://localhost:8501`
2. 创建新模型并配置参数
3. 上传数据集
4. 开始训练
5. 查看训练过程和结果
6. 使用模型进行预测

### 4. 使用 API
API 服务默认运行在 `http://localhost:8000`，提供以下主要端点：
- `GET /api/models`：获取所有模型列表
- `POST /api/models`：创建新模型
- `GET /api/models/{model_id}`：获取模型详情
- `POST /api/models/{model_id}/train`：训练模型
- `GET /api/models/{model_id}/train/status`：获取训练状态
- `POST /api/models/{model_id}/predict`：使用模型进行预测

## 注意事项

1. 确保在运行 Web 界面之前先启动 API 服务
2. 模型文件会保存在 `models` 目录下
3. 训练过程中可以随时查看训练状态和损失曲线

## 开发说明
- 项目使用 Python 3.8+ 开发 + Cursor 开发
- 主要依赖包括：numpy、fastapi、streamlit、scikit-learn