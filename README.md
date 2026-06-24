# 智盾·聊天诈骗识别系统

本项目基于 ChiFraud、豆包验证后的 DeepSeek 生成对话和公开网页弱标注数据，完成中文聊天诈骗数据构建、BERT/ChineseBERT 微调、Qwen3.7 RAG 评估与 Web 系统集成。

## 当前能力

- BERT、ChineseBERT 本地推理
- Qwen3.7-plus 云端 RAG 推理与结果校准
- 登录、注册、图形验证码和修改密码
- 文本、TXT、多轮对话与截图 OCR 输入
- 检测历史、原始输入回溯、反诈警示和模型中心
- SQLite 用户与检测记录持久化

## 快速启动

```cmd
cd /d D:\third2\comprehensive\Baseline\ChiFraud-main\app_v2
D:\anaconda\envs\nlp\python.exe -B web_app.py --host 127.0.0.1 --port 7871
```

访问 `http://127.0.0.1:7871`。局域网运行可使用 `start_app_lan.cmd`。

## 环境配置

```cmd
conda create -n nlp python=3.10 -y
conda activate nlp
python -m pip install -r requirements.txt
```

截图 OCR 还需要安装 Tesseract OCR，并配置中文语言包。

## Qwen API

Qwen3.7 RAG 需要通过环境变量配置 DashScope API Key，源码中不保存密钥：

```cmd
set DASHSCOPE_API_KEY=你的APIKey
```

也可使用 `QWEN_API_KEY`。BERT 和 ChineseBERT 不需要 API Key。选择 Qwen3.7 RAG 时，待检测文本会经过脱敏后发送至 DashScope，请勿提交不应上传云端的敏感信息。

## 数据集

`dataset/dialogue_binary_refined/`：

| 划分 | 总数 | 正常 | 诈骗 | 来源 |
| --- | ---: | ---: | ---: | --- |
| Train | 1360 | 1020 | 340 | ChiFraud 640、豆包验证对话 320、爬虫弱标注 400 |
| Dev | 120 | 90 | 30 | ChiFraud 2022 80、豆包验证对话 40 |
| Test | 120 | 90 | 30 | ChiFraud 2023 80、豆包验证对话 40 |

最终评估集 `dataset/dialogue_binary_refined_large_test/test.tsv` 共 500 条，其中正常 375 条、诈骗 125 条。数据来自 ChiFraud 2023 的 460 条样本和豆包验证对话的 40 条留出样本，不包含爬虫弱标注数据。

## 最终结果

| 模型 | 有效样本 | Accuracy | 诈骗 Precision | 诈骗 Recall | 诈骗 F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BERT refined | 500/500 | 96.60% | 91.54% | 95.20% | 93.33% |
| ChineseBERT refined | 500/500 | 95.80% | 90.00% | 93.60% | 91.76% |
| Qwen3.7 zero-shot | 495/500 | 88.28% | 68.33% | 99.19% | 80.92% |
| Qwen3.7 RAG + 校准 | 490/500 | 93.88% | 93.40% | 81.15% | 86.84% |
| Qwen3.7 RAG + 爬虫库 + 校准 | 485/500 | 97.53% | 94.40% | 95.93% | 95.16% |

Qwen 指标只统计 API 成功返回的有效样本。失败主要来自云端内容安全拦截和网络中断，因此不能与 500/500 完成推理的本地模型完全等同比较。

脱敏后的统一指标文件见 `results/latest_model_metrics.json`。

## 主要目录

```text
ChiFraud-main/
├─ app_v2/                                  # 当前 Web 应用
│  ├─ web/                                  # 前端页面、样式和交互
│  ├─ web_app.py                            # HTTP 服务、认证、历史和 OCR
│  └─ inference_backend.py                  # 统一推理后端
├─ dataset/
│  ├─ validated_data.jsonl
│  ├─ dialogue_binary_refined/
│  └─ dialogue_binary_refined_large_test/
├─ build_refined_dataset.py
├─ build_large_test_set.py
├─ evaluate_local_model.py
├─ evaluate_qwen_api.py
├─ evaluate_qwen_rag.py
├─ run.py
└─ train_eval.py
```

## GitHub 文件策略

仓库保留关键代码、紧凑数据划分、元数据、评估摘要和项目文档。API Key、`.env`、预训练模型、微调权重、大型原始导出、预测明细、数据库、日志与压缩包不提交。模型权重需通过课程交付包或其他大文件存储方式单独提供。
