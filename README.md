# 智盾·聊天诈骗识别系统

本项目是一个面向网络聊天场景的诈骗文本识别系统，基于 ChiFraud baseline 数据和本项目生成的诈骗对话数据完成数据整合、模型训练、评估和网页集成。

系统包含：

- BERT 本地分类模型
- ChineseBERT 本地分类模型
- Qwen API 云端大模型对比
- 前端工作台 `app_v2/web/`
- 后端推理服务 `app_v2/web_app.py`
- 数据处理、训练和评估脚本

## 快速启动

推荐使用已经配置好的 `nlp` 环境：

```cmd
cd /d D:\third2\comprehensive\Baseline\ChiFraud-main\app_v2
D:\anaconda\envs\nlp\python.exe -B web_app.py --host 127.0.0.1 --port 7871
```

启动后访问：

```text
http://127.0.0.1:7871
```

局域网演示可以使用 `--host 0.0.0.0` 启动，然后让同一局域网设备访问 `http://你的IPv4地址:7871`。

## 环境配置

```cmd
D:\anaconda\envs\nlp\python.exe -m pip install -r requirements.txt
```

截图 OCR 是可选功能，如果需要使用，请额外安装 `Pillow`、`pytesseract` 和 Tesseract OCR 主程序。

## Qwen API 配置

BERT 和 ChineseBERT 是本地模型，不需要 API Key。Qwen API 需要在启动服务前设置环境变量：

```cmd
set DASHSCOPE_API_KEY=你的APIKey
```

或：

```cmd
set QWEN_API_KEY=你的APIKey
```

本仓库不会在源码中写死 API Key。

## 数据说明

- `dataset/data.jsonl`：大模型生成的诈骗对话数据
- `dataset/dialogue_binary_matched_1x/`：匹配采样 1x 数据集
- `dataset/dialogue_binary_matched_2x/`：匹配采样 2x 数据集
- `app_v2/dataset/`：网页演示和推理所需的小规模数据划分

原始 ChiFraud 大 CSV 文件和完整中间转换文件体积较大，未纳入 GitHub 版本库。

## 模型结果

以 `dialogue_binary_matched_2x/test.tsv` 为测试集，共 120 条样本，其中正常 90 条，诈骗 30 条。

| 模型 | Accuracy | 诈骗 Precision | 诈骗 Recall | 诈骗 F1 |
| --- | ---: | ---: | ---: | ---: |
| BERT | 91.67% | 81.25% | 86.67% | 83.87% |
| ChineseBERT | 95.00% | 90.00% | 90.00% | 90.00% |
| Qwen API | 87.50% | 67.44% | 96.67% | 79.45% |

结论：ChineseBERT 综合表现最好，BERT 适合作为轻量备用，Qwen API 适合做大模型对比展示。

## 主要目录

```text
ChiFraud-main/
|- app_v2/                  # 当前网页应用版本
|  |- web/                  # 前端页面、样式和交互脚本
|  |- web_app.py            # 本地 HTTP 服务入口
|  |- inference_backend.py  # 统一推理后端
|  |- models/               # 模型结构
|  `- dataset/              # 网页演示数据
|- dataset/                 # 数据处理结果和实验数据
|- run.py                   # baseline 训练入口
|- train_eval.py            # 训练和评估逻辑
`- evaluate_qwen_api.py     # Qwen API 评估脚本
```

## GitHub 上传说明

本仓库保留项目关键代码、网页前端、后端推理服务、训练与评估脚本，以及小规模关键数据划分。

以下内容不会上传到 GitHub：

- 预训练模型权重：`pretrained/`、`app_v2/pretrained/`
- 微调后的模型权重：`saved_dict/`、`app_v2/saved_dict/`
- 本地数据库、日志、运行结果和压缩包
- Qwen / DashScope API Key
- 原始 ChiFraud 大 CSV 和完整中间转换文件

如需使用 Qwen API，请在本地启动前设置环境变量。

## 常见问题

### 页面能打开，但点击检测失败

先查看启动后端的终端输出。常见原因包括依赖未安装、模型文件缺失、端口被占用或 Qwen API Key 未配置。

### 第一次检测很慢

模型是懒加载的，第一次选择某个模型时会加载权重，后续预测会快很多。
