# 智盾·聊天诈骗识别系统 应用包说明

这是一个可本地运行的网页应用包，包含前端页面、后端推理服务、BERT/ChineseBERT 权重和必要的模型代码。

## 启动方式

本机演示：

```cmd
start_app.cmd
```

浏览器打开：

```text
http://127.0.0.1:7871
```

局域网演示：

```cmd
start_app_lan.cmd
```

然后在本机运行 `ipconfig` 查看 IPv4 地址，其他同一局域网设备访问：

```text
http://本机IPv4地址:7871
```

## 环境要求

推荐 Python 3.10。安装依赖：

```cmd
python -m pip install -r requirements.txt
```

如果使用本机已有 `nlp` 环境：

```cmd
D:\anaconda\envs\nlp\python.exe -m pip install -r requirements.txt
```

## 模型说明

- BERT：本地离线推理，速度快。
- ChineseBERT：本地离线推理，当前推荐默认模型。
- Qwen API：云端千问接口，不占用本机显存，但需要网络和 API Key。

Qwen API 启动前需要设置：

```cmd
set DASHSCOPE_API_KEY=你的千问APIKey
```

如果不设置，BERT 和 ChineseBERT 仍可正常使用。

## 当前测试集结果

测试集：`dataset/dialogue_binary_matched_2x/test.tsv`，共 120 条。

| 模型 | Acc | 诈骗类 F1 |
| --- | ---: | ---: |
| BERT | 91.67% | 83.87% |
| ChineseBERT | 95.00% | 90.00% |
| Qwen API | 87.50% | 79.45% |
