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

- BERT：加载 `ChiFraudDialogRefined` 权重进行本地离线推理。
- ChineseBERT：加载 `ChiFraudDialogRefined` 权重并融合字形、拼音特征。
- Qwen3.7 RAG：调用 `qwen3.7-plus`，结合 refined 训练集、爬虫语料检索和校准规则。

Qwen API 启动前需要设置：

```cmd
set DASHSCOPE_API_KEY=你的千问APIKey
```

如果不设置，BERT 和 ChineseBERT 仍可正常使用。

## 最终大测试集结果

测试集：`dataset/dialogue_binary_refined_large_test/test.tsv`，共 500 条，正常 375 条、诈骗 125 条。

| 模型 | 有效样本 | Acc | 诈骗类 F1 |
| --- | ---: | ---: | ---: |
| BERT refined | 500/500 | 96.60% | 93.33% |
| ChineseBERT refined | 500/500 | 95.80% | 91.76% |
| Qwen3.7 RAG + 爬虫库 + 校准 | 485/500 | 97.53% | 95.16% |

Qwen 结果仅统计 API 成功返回的 485 条样本。
