import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import time
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# 设置页面配置
st.set_page_config(
    page_title="MLP 神经网络训练与推理",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API 基础 URL
API_BASE_URL = "http://localhost:8000/api"

# 添加自定义 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4257b2;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #3c9d9b;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d1f0e0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stProgress > div > div > div > div {
        background-color: #4257b2;
    }
</style>
""", unsafe_allow_html=True)

# 初始化会话状态
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'current_model_id' not in st.session_state:
    st.session_state.current_model_id = None
if 'training_status' not in st.session_state:
    st.session_state.training_status = None
if 'loss_history' not in st.session_state:
    st.session_state.loss_history = {"train": [], "val": []}
if 'data_uploaded' not in st.session_state:
    st.session_state.data_uploaded = False
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'target_name' not in st.session_state:
    st.session_state.target_name = None
if 'is_classification' not in st.session_state:
    st.session_state.is_classification = True
if 'classes' not in st.session_state:
    st.session_state.classes = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'probabilities' not in st.session_state:
    st.session_state.probabilities = None

# API 函数
def fetch_models():
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"获取模型列表失败: {response.text}")
            return []
    except Exception as e:
        st.error(f"API 连接错误: {str(e)}")
        return []

def create_model(name, model_type, config):
    try:
        payload = {
            "name": name,
            "type": model_type,
            "config": config
        }
        response = requests.post(f"{API_BASE_URL}/models", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"创建模型失败: {response.text}")
            return None
    except Exception as e:
        st.error(f"API 连接错误: {str(e)}")
        return None

def get_model_details(model_id):
    try:
        response = requests.get(f"{API_BASE_URL}/models/{model_id}")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"获取模型详情失败: {response.text}")
            return None
    except Exception as e:
        st.error(f"API 连接错误: {str(e)}")
        return None

def train_model(model_id, X, y, training_params=None):
    try:
        # Convert inputs to numpy arrays
        # X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        # Handle target shape
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        payload = {
            "data": {
                "X": X.tolist() if isinstance(X, np.ndarray) else X,
                "y": y.tolist()
            }
        }
        if training_params:
            payload["training_params"] = training_params
        
        print(payload)
        response = requests.post(f"{API_BASE_URL}/models/{model_id}/train", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"启动训练失败: {response.text}")
            return None
    except Exception as e:
        st.error(f"API 连接错误: {str(e)}")
        return None

def get_training_status(model_id):
    try:
        response = requests.get(f"{API_BASE_URL}/models/{model_id}/train/status")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"获取训练状态失败: {response.text}")
            return None
    except Exception as e:
        st.error(f"API 连接错误: {str(e)}")
        return None

def predict(model_id, X):
    try:
        payload = {
            "data": X.tolist() if isinstance(X, np.ndarray) else X
        }
        response = requests.post(f"{API_BASE_URL}/models/{model_id}/predict", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"预测失败: {response.text}")
            return None
    except Exception as e:
        st.error(f"API 连接错误: {str(e)}")
        return None

def delete_model(model_id):
    try:
        response = requests.delete(f"{API_BASE_URL}/models/{model_id}")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"删除模型失败: {response.text}")
            return None
    except Exception as e:
        st.error(f"API 连接错误: {str(e)}")
        return None

# 辅助函数
def plot_loss_history():
    if not st.session_state.loss_history["train"]:
        return None
    
    fig = go.Figure()
    
    # 添加训练损失曲线
    fig.add_trace(go.Scatter(
        x=list(range(len(st.session_state.loss_history["train"]))),
        y=st.session_state.loss_history["train"],
        mode='lines',
        name='训练损失',
        line=dict(color='blue', width=2)
    ))
    
    # 如果有验证损失，也添加进去
    if st.session_state.loss_history["val"]:
        fig.add_trace(go.Scatter(
            x=list(range(len(st.session_state.loss_history["val"]))),
            y=st.session_state.loss_history["val"],
            mode='lines',
            name='验证损失',
            line=dict(color='red', width=2)
        ))
    
    fig.update_layout(
        title='训练过程损失曲线',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        legend_title='数据集',
        template='plotly_white',
        height=400
    )
    
    return fig

def plot_confusion_matrix(y_true, y_pred, classes=None):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    if classes is None:
        classes = [f"类别 {i}" for i in range(cm.shape[0])]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.title('混淆矩阵')
    
    # 将图转换为 Streamlit 可显示的格式
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    
    return buf

def plot_feature_importance(model, feature_names):
    # 这个函数需要根据您的 MLP 实现来调整
    # 这里只是一个示例，假设模型有权重可以访问
    if hasattr(model, 'network') and model.network.layers:
        # 获取第一层的权重
        weights = np.abs(model.network.layers[0].weights)
        importance = np.mean(weights, axis=1)
        
        fig = px.bar(
            x=importance,
            y=feature_names,
            orientation='h',
            title='特征重要性（基于第一层权重）',
            labels={'x': '重要性', 'y': '特征'}
        )
        fig.update_layout(height=400)
        return fig
    return None

def preprocess_data(df, target_column, test_size=0.2, random_state=42, normalize=True):
    # 分离特征和目标
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # 保存特征名和目标名
    feature_names = X.columns.tolist()
    
    # 检查是否为分类问题
    unique_values = y.unique()
    is_classification = len(unique_values) < 10 or y.dtype == 'object'
    
    # 如果是分类问题，将目标转换为数值
    if is_classification:
        if y.dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)
            classes = le.classes_
        else:
            classes = unique_values
    else:
        classes = None
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 标准化特征（如果需要）
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        X_train = X_train.values
        X_test = X_test.values
    
    return X_train, X_test, y_train, y_test, feature_names, is_classification, classes

# 主应用界面
def main():
    st.markdown('<h1 class="main-header">MLP 神经网络训练与推理平台</h1>', unsafe_allow_html=True)
    
    # 侧边栏 - 模型管理
    with st.sidebar:
        st.markdown('<h2 class="section-header">模型管理</h2>', unsafe_allow_html=True)
        
        # 刷新模型列表
        if st.button("刷新模型列表"):
            models = fetch_models()
            if models:
                st.session_state.models = {model["model_id"]: model for model in models}
        
        # 显示现有模型
        if st.session_state.models:
            model_options = ["选择模型..."] + [f"{model['name']} ({model_id})" for model_id, model in st.session_state.models.items()]
            selected_model = st.selectbox("选择模型", model_options)
            
            if selected_model != "选择模型...":
                model_id = selected_model.split("(")[-1].strip(")")
                st.session_state.current_model_id = model_id
                
                # 显示模型详情
                model_details = get_model_details(model_id)
                if model_details:
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.write(f"**模型类型:** {model_details['type']}")
                    st.write(f"**状态:** {model_details['status']}")
                    st.write(f"**创建时间:** {model_details['created_at']}")
                    
                    with st.expander("模型配置"):
                        st.json(model_details['config'])
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # 删除模型按钮
                    if st.button("删除模型"):
                        if delete_model(model_id):
                            st.success(f"模型 {model_id} 已删除")
                            # 更新模型列表
                            if model_id in st.session_state.models:
                                del st.session_state.models[model_id]
                            st.session_state.current_model_id = None
                            st.rerun()
                            # st.experimental_rerun()
        
        st.markdown('<h2 class="section-header">创建新模型</h2>', unsafe_allow_html=True)
        
        # 创建新模型表单
        with st.form("create_model_form"):
            model_name = st.text_input("模型名称", "my_mlp_model")
            model_type = st.selectbox("模型类型", ["classifier", "regressor"])
            
            st.markdown("**网络结构**")
            hidden_layers = st.text_input("隐藏层大小 (逗号分隔)", "100,50")
            activation = st.selectbox("激活函数", ["relu", "sigmoid", "tanh", "leaky_relu"])
            
            st.markdown("**优化器设置**")
            solver = st.selectbox("优化器", ["adam", "sgd", "momentum", "rmsprop"])
            learning_rate = st.number_input("学习率", min_value=0.0001, max_value=1.0, value=0.001, format="%.4f")
            batch_size = st.number_input("批量大小", min_value=1, max_value=1000, value=32)
            max_iter = st.number_input("最大迭代次数", min_value=10, max_value=10000, value=200)
            
            st.markdown("**正则化设置**")
            reg_type = st.selectbox("正则化类型", ["none", "l1", "l2", "elastic_net"])
            alpha = st.number_input("正则化强度", min_value=0.0, max_value=1.0, value=0.0001, format="%.5f")
            
            st.markdown("**其他设置**")
            validation_split = st.slider("验证集比例", min_value=0.0, max_value=0.5, value=0.2, step=0.05)
            early_stopping = st.checkbox("启用早停", value=True)
            n_iter_no_change = st.number_input("早停容忍次数", min_value=1, max_value=100, value=10)
            
            use_parallel = st.checkbox("启用并行训练", value=False)
            num_threads = st.number_input("线程数", min_value=1, max_value=16, value=4, disabled=not use_parallel)
            
            submit_button = st.form_submit_button("创建模型")
            
            if submit_button:
                # 解析隐藏层配置
                hidden_layer_sizes = [int(x.strip()) for x in hidden_layers.split(",")]
                
                # 创建模型配置
                config = {
                    "hidden_layer_sizes": hidden_layer_sizes,
                    "activation": activation,
                    "init_type": "he" if activation in ["relu", "leaky_relu"] else "xavier",
                    "solver": solver,
                    "alpha": alpha,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "max_iter": max_iter,
                    "validation_split": validation_split,
                    "early_stopping": early_stopping,
                    "n_iter_no_change": n_iter_no_change,
                    "reg_type": reg_type,
                    "use_parallel": use_parallel,
                    "num_threads": num_threads
                }
                
                # 调用 API 创建模型
                result = create_model(model_name, model_type, config)
                if result:
                    st.success(f"模型创建成功! 模型 ID: {result['model_id']}")
                    # 更新模型列表
                    st.session_state.models[result['model_id']] = result
                    st.session_state.current_model_id = result['model_id']
                    st.rerun()
    
    # 主界面 - 标签页
    tab1, tab2, tab3, tab4 = st.tabs(["数据管理", "模型训练", "模型评估", "模型推理"])
    
    # 数据管理标签页
    with tab1:
        st.markdown('<h2 class="section-header">数据上传与预处理</h2>', unsafe_allow_html=True)
        
        # 数据上传
        uploaded_file = st.file_uploader("上传数据集 (CSV 格式)", type=["csv"])
        
        if uploaded_file is not None:
            # 读取数据
            df = pd.read_csv(uploaded_file)
            st.write("数据预览:")
            st.dataframe(df.head())
            
            # 数据预处理选项
            st.markdown('<h3 class="section-header">数据预处理</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                target_column = st.selectbox("选择目标列", df.columns)
            with col2:
                test_size = st.slider("测试集比例", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
            
            normalize = st.checkbox("标准化特征", value=True)
            
            if st.button("处理数据"):
                with st.spinner("正在处理数据..."):
                    X_train, X_test, y_train, y_test, feature_names, is_classification, classes = preprocess_data(
                        df, target_column, test_size, normalize=normalize
                    )
                    
                    # 保存到会话状态
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.feature_names = feature_names
                    st.session_state.target_name = target_column
                    st.session_state.is_classification = is_classification
                    st.session_state.classes = classes
                    st.session_state.data_uploaded = True
                
                st.success("数据处理完成!")
                st.write(f"训练集: {X_train.shape[0]} 样本, 测试集: {X_test.shape[0]} 样本")
                st.write(f"特征数量: {X_train.shape[1]}")
                st.write(f"问题类型: {'分类' if is_classification else '回归'}")
                
                if is_classification:
                    st.write(f"类别数量: {len(classes)}")
                    st.write("类别分布:")
                    
                    # 显示类别分布
                    y_train_df = pd.DataFrame({'类别': y_train})
                    class_counts = y_train_df['类别'].value_counts().reset_index()
                    class_counts.columns = ['类别', '数量']
                    
                    fig = px.bar(
                        class_counts, 
                        x='类别', 
                        y='数量',
                        title='训练集类别分布',
                        color='数量',
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig)
    
    # 模型训练标签页
    with tab2:
        st.markdown('<h2 class="section-header">模型训练</h2>', unsafe_allow_html=True)
        
        if not st.session_state.current_model_id:
            st.warning("请先在侧边栏选择或创建一个模型")
        elif not st.session_state.data_uploaded:
            st.warning("请先在数据管理标签页上传并处理数据")
        else:
            model_id = st.session_state.current_model_id
            model_details = get_model_details(model_id)
            
            if model_details:
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.write(f"**当前模型:** {model_details['name']} ({model_id})")
                st.write(f"**模型类型:** {model_details['type']}")
                st.write(f"**当前状态:** {model_details['status']}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # 检查模型类型与数据类型是否匹配
                if (model_details['type'] == 'classifier' and not st.session_state.is_classification) or \
                   (model_details['type'] == 'regressor' and st.session_state.is_classification):
                    st.error(f"模型类型 ({model_details['type']}) 与数据类型 ({'分类' if st.session_state.is_classification else '回归'}) 不匹配!")
                else:
                    # 训练参数调整
                    with st.expander("训练参数调整 (可选)"):
                        batch_size = st.number_input("批量大小", min_value=1, max_value=1000, value=model_details['config'].get('batch_size', 32))
                        learning_rate = st.number_input("学习率", min_value=0.0001, max_value=1.0, value=model_details['config'].get('learning_rate', 0.001), format="%.4f")
                        max_iter = st.number_input("最大迭代次数", min_value=10, max_value=10000, value=model_details['config'].get('max_iter', 200))
                        
                        training_params = {
                            "batch_size": batch_size,
                            "learning_rate": learning_rate,
                            "max_iter": max_iter
                        }
                    
                    # 开始训练按钮
                    if st.button("开始训练"):
                        with st.spinner("正在启动训练..."):
                            print(type(st.session_state.y_train))
                            result = train_model(
                                model_id, 
                                st.session_state.X_train, 
                                st.session_state.y_train, 
                                training_params
                            )
                            
                            if result:
                                st.success("训练已启动!")
                                st.session_state.training_status = "training_started"
                                st.session_state.loss_history = {"train": [], "val": []}
                    
                    # 训练状态监控
                    if st.session_state.training_status:
                        st.markdown('<h3 class="section-header">训练进度</h3>', unsafe_allow_html=True)
                        
                        status_placeholder = st.empty()
                        progress_bar = st.progress(0)
                        loss_chart_placeholder = st.empty()
                        
                        # 轮询训练状态
                        while True:
                            status = get_training_status(model_id)
                            
                            if not status:
                                status_placeholder.error("无法获取训练状态")
                                break
                            
                            status_placeholder.markdown(f"<div class='info-box'>当前状态: {status['status']}</div>", unsafe_allow_html=True)
                            
                            # 更新损失历史
                            if 'loss_history' in status:
                                st.session_state.loss_history = status['loss_history']
                                
                                # 更新进度条
                                if 'train' in status['loss_history'] and status['loss_history']['train']:
                                    progress = min(len(status['loss_history']['train']) / max_iter, 1.0)
                                    progress_bar.progress(progress)
                                
                                # 更新损失图表
                                fig = plot_loss_history()
                                if fig:
                                    loss_chart_placeholder.plotly_chart(fig)
                            
                            # 检查训练是否完成
                            if status['status'] in ['completed', 'failed']:
                                if status['status'] == 'completed':
                                    status_placeholder.markdown("<div class='success-box'>训练完成!</div>", unsafe_allow_html=True)
                                else:
                                    status_placeholder.markdown(f"<div class='error-box'>训练失败: {status.get('error', '未知错误')}</div>", unsafe_allow_html=True)
                                
                                st.session_state.training_status = status['status']
                                break
                            
                            time.sleep(2)  # 每2秒轮询一次
    
    # 模型评估标签页
    with tab3:
        st.markdown('<h2 class="section-header">模型评估</h2>', unsafe_allow_html=True)
        
        if not st.session_state.current_model_id:
            st.warning("请先在侧边栏选择或创建一个模型")
        elif not st.session_state.data_uploaded:
            st.warning("请先在数据管理标签页上传并处理数据")
        else:
            model_id = st.session_state.current_model_id
            model_details = get_model_details(model_id)
            
            if model_details and model_details.get('is_fitted', False):
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.write(f"**当前模型:** {model_details['name']} ({model_id})")
                st.markdown('</div>', unsafe_allow_html=True)
                
                if st.button("评估模型"):
                    with st.spinner("正在评估模型..."):
                        # 获取测试集预测
                        result = predict(model_id, st.session_state.X_test)
                        
                        if result:
                            if 'predictions' in result:
                                predictions = np.array(result['predictions'])
                                st.session_state.predictions = predictions
                                
                                if 'probabilities' in result:
                                    probabilities = np.array(result['probabilities'])
                                    st.session_state.probabilities = probabilities
                                
                                # 显示评估结果
                                st.markdown('<h3 class="section-header">评估结果</h3>', unsafe_allow_html=True)
                                
                                if st.session_state.is_classification:
                                    # 分类评估指标
                                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                                    
                                    accuracy = accuracy_score(st.session_state.y_test, predictions)
                                    
                                    if len(np.unique(st.session_state.y_test)) > 2:
                                        # 多分类
                                        precision = precision_score(st.session_state.y_test, predictions, average='macro')
                                        recall = recall_score(st.session_state.y_test, predictions, average='macro')
                                        f1 = f1_score(st.session_state.y_test, predictions, average='macro')
                                    else:
                                        # 二分类
                                        precision = precision_score(st.session_state.y_test, predictions)
                                        recall = recall_score(st.session_state.y_test, predictions)
                                        f1 = f1_score(st.session_state.y_test, predictions)
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("准确率", f"{accuracy:.4f}")
                                        st.metric("精确率", f"{precision:.4f}")
                                    with col2:
                                        st.metric("召回率", f"{recall:.4f}")
                                        st.metric("F1 分数", f"{f1:.4f}")
                                    
                                    # 混淆矩阵
                                    st.subheader("混淆矩阵")
                                    cm_buf = plot_confusion_matrix(
                                        st.session_state.y_test, 
                                        predictions,
                                        st.session_state.classes
                                    )
                                    st.image(cm_buf)
                                    
                                    # ROC 曲线 (仅适用于二分类)
                                    if len(np.unique(st.session_state.y_test)) == 2 and 'probabilities' in result:
                                        from sklearn.metrics import roc_curve, auc
                                        
                                        st.subheader("ROC 曲线")
                                        probas = np.array(result['probabilities'])
                                        if probas.shape[1] == 2:  # 确保是二分类概率
                                            fpr, tpr, _ = roc_curve(st.session_state.y_test, probas[:, 1])
                                            roc_auc = auc(fpr, tpr)
                                            
                                            fig = px.area(
                                                x=fpr, y=tpr,
                                                title=f'ROC 曲线 (AUC = {roc_auc:.4f})',
                                                labels=dict(x='假正例率', y='真正例率'),
                                                width=700, height=500
                                            )
                                            fig.add_shape(
                                                type='line', line=dict(dash='dash'),
                                                x0=0, x1=1, y0=0, y1=1
                                            )
                                            
                                            st.plotly_chart(fig)
                                else:
                                    # 回归评估指标
                                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                                    
                                    mse = mean_squared_error(st.session_state.y_test, predictions)
                                    rmse = np.sqrt(mse)
                                    mae = mean_absolute_error(st.session_state.y_test, predictions)
                                    r2 = r2_score(st.session_state.y_test, predictions)
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("均方误差 (MSE)", f"{mse:.4f}")
                                        st.metric("均方根误差 (RMSE)", f"{rmse:.4f}")
                                    with col2:
                                        st.metric("平均绝对误差 (MAE)", f"{mae:.4f}")
                                        st.metric("决定系数 (R²)", f"{r2:.4f}")
                                    
                                    # 预测值与真实值对比图
                                    st.subheader("预测值与真实值对比")
                                    
                                    fig = px.scatter(
                                        x=st.session_state.y_test.flatten(), 
                                        y=predictions.flatten(),
                                        labels={'x': '真实值', 'y': '预测值'},
                                        title='预测值与真实值对比'
                                    )
                                    
                                    # 添加对角线
                                    min_val = min(st.session_state.y_test.min(), predictions.min())
                                    max_val = max(st.session_state.y_test.max(), predictions.max())
                                    fig.add_shape(
                                        type='line',
                                        x0=min_val, y0=min_val,
                                        x1=max_val, y1=max_val,
                                        line=dict(color='red', dash='dash')
                                    )
                                    
                                    st.plotly_chart(fig)
                                    
                                    # 残差图
                                    st.subheader("残差图")
                                    residuals = st.session_state.y_test.flatten() - predictions.flatten()
                                    
                                    fig = px.scatter(
                                        x=predictions.flatten(),
                                        y=residuals,
                                        labels={'x': '预测值', 'y': '残差'},
                                        title='残差图'
                                    )
                                    
                                    # 添加水平线
                                    fig.add_shape(
                                        type='line',
                                        x0=min(predictions.flatten()),
                                        y0=0,
                                        x1=max(predictions.flatten()),
                                        y1=0,
                                        line=dict(color='red', dash='dash')
                                    )
                                    
                                    st.plotly_chart(fig)
                                
                                # 特征重要性（如果可用）
                                if hasattr(model_details, 'network') and len(st.session_state.feature_names) > 0:
                                    st.subheader("特征重要性")
                                    fig = plot_feature_importance(model_details, st.session_state.feature_names)
                                    if fig:
                                        st.plotly_chart(fig)
                                    else:
                                        st.info("无法计算特征重要性")
    
    # 模型推理标签页
    with tab4:
        st.markdown('<h2 class="section-header">模型推理</h2>', unsafe_allow_html=True)
        
        if not st.session_state.current_model_id:
            st.warning("请先在侧边栏选择或创建一个模型")
        elif not st.session_state.data_uploaded:
            st.warning("请先在数据管理标签页上传并处理数据")
        else:
            model_id = st.session_state.current_model_id
            model_details = get_model_details(model_id)
            
            if model_details and model_details.get('is_fitted', False):
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.write(f"**当前模型:** {model_details['name']} ({model_id})")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # 选择推理方式
                inference_mode = st.radio("选择推理方式", ["使用测试集样本", "手动输入特征值"])
                
                if inference_mode == "使用测试集样本":
                    # 从测试集选择样本
                    if st.session_state.X_test is not None and st.session_state.X_test.shape[0] > 0:
                        sample_index = st.slider("选择测试集样本索引", 0, st.session_state.X_test.shape[0] - 1, 0)
                        
                        # 显示样本特征
                        st.subheader("样本特征")
                        sample_features = st.session_state.X_test[sample_index]
                        
                        feature_df = pd.DataFrame({
                            '特征名': st.session_state.feature_names,
                            '特征值': sample_features
                        })
                        st.dataframe(feature_df)
                        
                        # 显示真实标签
                        true_label = st.session_state.y_test[sample_index]
                        if st.session_state.is_classification:
                            if st.session_state.classes is not None:
                                class_idx = int(true_label) if isinstance(true_label, (int, float)) else true_label
                                if isinstance(class_idx, np.ndarray):
                                    class_idx = np.argmax(class_idx)
                                true_label_name = st.session_state.classes[class_idx]
                                st.write(f"**真实标签:** {true_label_name} (类别 {class_idx})")
                            else:
                                st.write(f"**真实标签:** {true_label}")
                        else:
                            st.write(f"**真实值:** {true_label}")
                        
                        # 进行预测
                        if st.button("进行预测"):
                            with st.spinner("正在预测..."):
                                # 准备单个样本
                                X_sample = sample_features.reshape(1, -1)
                                
                                # 调用 API 进行预测
                                result = predict(model_id, X_sample)
                                
                                if result:
                                    st.success("预测完成!")
                                    
                                    # 显示预测结果
                                    if 'predictions' in result:
                                        predictions = np.array(result['predictions'])
                                        
                                        if st.session_state.is_classification:
                                            # 分类问题
                                            pred_class = predictions[0]
                                            if isinstance(pred_class, np.ndarray):
                                                pred_class = np.argmax(pred_class)
                                            
                                            if st.session_state.classes is not None:
                                                pred_class_name = st.session_state.classes[int(pred_class)]
                                                st.markdown(f"<div class='success-box'><h3>预测标签: {pred_class_name} (类别 {int(pred_class)})</h3></div>", unsafe_allow_html=True)
                                            else:
                                                st.markdown(f"<div class='success-box'><h3>预测标签: {pred_class}</h3></div>", unsafe_allow_html=True)
                                            
                                            # 显示概率分布（如果有）
                                            if 'probabilities' in result:
                                                probabilities = np.array(result['probabilities'])[0]
                                                
                                                if len(probabilities) > 1:
                                                    st.subheader("类别概率分布")
                                                    
                                                    proba_df = pd.DataFrame({
                                                        '类别': [st.session_state.classes[i] if st.session_state.classes is not None else f"类别 {i}" for i in range(len(probabilities))],
                                                        '概率': probabilities
                                                    })
                                                    
                                                    fig = px.bar(
                                                        proba_df,
                                                        x='类别',
                                                        y='概率',
                                                        title='预测概率分布',
                                                        color='概率',
                                                        color_continuous_scale='Viridis'
                                                    )
                                                    st.plotly_chart(fig)
                                        else:
                                            # 回归问题
                                            pred_value = predictions[0]
                                            if isinstance(pred_value, np.ndarray):
                                                pred_value = pred_value[0]
                                            
                                            st.markdown(f"<div class='success-box'><h3>预测值: {pred_value:.4f}</h3></div>", unsafe_allow_html=True)
                                            
                                            # 显示真实值与预测值的对比
                                            st.subheader("真实值与预测值对比")
                                            
                                            compare_df = pd.DataFrame({
                                                '类型': ['真实值', '预测值'],
                                                '值': [float(true_label), float(pred_value)]
                                            })
                                            
                                            fig = px.bar(
                                                compare_df,
                                                x='类型',
                                                y='值',
                                                title='真实值与预测值对比',
                                                color='类型'
                                            )
                                            st.plotly_chart(fig)
                    else:
                        st.error("测试集为空，无法选择样本")
                
                else:  # 手动输入特征值
                    st.subheader("手动输入特征值")
                    
                    # 创建输入表单
                    with st.form("manual_input_form"):
                        feature_values = []
                        
                        # 根据特征名创建输入字段
                        if st.session_state.feature_names:
                            for feature_name in st.session_state.feature_names:
                                feature_values.append(
                                    st.number_input(f"{feature_name}", value=0.0, format="%.4f")
                                )
                        else:
                            # 如果没有特征名，则使用索引
                            n_features = st.session_state.X_train.shape[1] if st.session_state.X_train is not None else 0
                            for i in range(n_features):
                                feature_values.append(
                                    st.number_input(f"特征 {i+1}", value=0.0, format="%.4f")
                                )
                        
                        submit_button = st.form_submit_button("进行预测")
                    
                    if submit_button:
                        with st.spinner("正在预测..."):
                            # 准备特征数据
                            X_sample = np.array(feature_values).reshape(1, -1)
                            
                            # 调用 API 进行预测
                            result = predict(model_id, X_sample)
                            
                            if result:
                                st.success("预测完成!")
                                
                                # 显示预测结果
                                if 'predictions' in result:
                                    predictions = np.array(result['predictions'])
                                    
                                    if st.session_state.is_classification:
                                        # 分类问题
                                        pred_class = predictions[0]
                                        if isinstance(pred_class, np.ndarray):
                                            pred_class = np.argmax(pred_class)
                                        
                                        if st.session_state.classes is not None:
                                            pred_class_name = st.session_state.classes[int(pred_class)]
                                            st.markdown(f"<div class='success-box'><h3>预测标签: {pred_class_name} (类别 {int(pred_class)})</h3></div>", unsafe_allow_html=True)
                                        else:
                                            st.markdown(f"<div class='success-box'><h3>预测标签: {pred_class}</h3></div>", unsafe_allow_html=True)
                                        
                                        # 显示概率分布（如果有）
                                        if 'probabilities' in result:
                                            probabilities = np.array(result['probabilities'])[0]
                                            
                                            if len(probabilities) > 1:
                                                st.subheader("类别概率分布")
                                                
                                                proba_df = pd.DataFrame({
                                                    '类别': [st.session_state.classes[i] if st.session_state.classes is not None else f"类别 {i}" for i in range(len(probabilities))],
                                                    '概率': probabilities
                                                })
                                                
                                                fig = px.bar(
                                                    proba_df,
                                                    x='类别',
                                                    y='概率',
                                                    title='预测概率分布',
                                                    color='概率',
                                                    color_continuous_scale='Viridis'
                                                )
                                                st.plotly_chart(fig)
                                    else:
                                        # 回归问题
                                        pred_value = predictions[0]
                                        if isinstance(pred_value, np.ndarray):
                                            pred_value = pred_value[0]
                                        
                                        st.markdown(f"<div class='success-box'><h3>预测值: {pred_value:.4f}</h3></div>", unsafe_allow_html=True)
            else:
                st.warning("当前模型尚未训练，请先在模型训练标签页进行训练")

# 运行应用
if __name__ == "__main__":
    main()
