# 智盾·聊天诈骗识别系统 前端交付说明

这份项目包用于前端同学继续修改网页并联调后端模型。当前系统是一个 Python 本地 HTTP 服务，浏览器页面通过接口调用模型。

## 1. 项目入口

前端主要改这里：

```text
web/index.html
web/styles.css
web/app.js
```

后端接口入口：

```text
web_app.py
```

模型推理入口：

```text
inference_backend.py
```

## 2. 环境准备

推荐使用 Anaconda，新建或使用已有 `nlp` 环境。

如果从零创建环境：

```cmd
conda create -n nlp python=3.10 -y
conda activate nlp
```

安装依赖：

```cmd
cd /d D:\third2\comprehensive\Baseline\ChiFraud-main
python -m pip install -r requirements.txt
```

如果使用项目原来的环境，可以直接指定 Python：

```cmd
D:\anaconda\envs\nlp\python.exe -m pip install -r requirements.txt
```

如果机器有 NVIDIA GPU，建议确认 PyTorch 能识别 CUDA：

```cmd
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

如果输出 `True`，说明 GPU 可用；如果是 `False`，BERT/ChineseBERT 也可以用 CPU 跑，只是速度会慢。

## 3. 启动后端和网页

本机启动：

```cmd
cd /d D:\third2\comprehensive\Baseline\ChiFraud-main
python -B web_app.py --host 127.0.0.1 --port 7871
```

或者直接运行：

```cmd
start_app.cmd
```

浏览器打开：

```text
http://127.0.0.1:7871
```

局域网访问：

```cmd
start_app_lan.cmd
```

然后用 `ipconfig` 查看本机 IPv4，其他同一局域网电脑访问：

```text
http://本机IPv4地址:7871
```

如果访问失败，检查 Windows 防火墙是否允许 Python 或 7871 端口入站。

## 4. 后端接口

### 获取模型信息

```http
GET /api/models
```

### 健康检查

```http
GET /api/health
```

### 诈骗检测

```http
POST /api/predict
Content-Type: application/json
```

请求示例：

```json
{
  "model": "chinesebert",
  "messages": [
    {"role": "用户", "text": "我看到兼职刷单，需要先垫付 300 元。"},
    {"role": "对方", "text": "名额有限，现在转账才能进群。"}
  ]
}
```

也可以直接传一整段文本：

```json
{
  "model": "chinesebert",
  "text": "用户：我看到兼职刷单，需要先垫付 300 元。\n对方：名额有限，现在转账才能进群。"
}
```

可选模型：

```text
bert
chinesebert
qwen_api
```

## 5. 前端开发建议

改页面时不需要重新训练模型。保持请求接口 `/api/predict` 即可。

当前输入方式：

```text
1. 多轮对话输入框
2. 直接粘贴大段文本
3. 上传 TXT 文件
```

当前不是批处理模式。粘贴或上传的文本会作为“一条样本/一段聊天记录”检测。如果要一次上传很多条并逐条输出结果，需要新增批量接口或前端循环调用 `/api/predict`。

## 6. 模型说明

当前本地默认模型：

```text
ChineseBERT
```

最终大测试集结果：

```text
BERT refined（500/500）:
Acc 96.60%，诈骗类 F1 93.33%

ChineseBERT refined（500/500）:
Acc 95.80%，诈骗类 F1 91.76%

Qwen3.7 RAG + 爬虫库 + 校准（485/500）:
Acc 97.53%，诈骗类 F1 95.16%
```

网页里的 `qwen_api` 当前对应 Qwen3.7-plus 云端接口，并结合本地 TF-IDF RAG 检索与校准规则。Qwen 指标仅统计 API 成功返回的 485 条样本；本地 Qwen LoRA 仅保留为早期实验代码，不属于当前演示路线。

## 7. 常见问题

### 页面能打开，但检测失败

先看启动后端的终端输出。后端会打印 traceback。常见原因：

```text
1. 模型文件缺失
2. Python 环境依赖没装全
3. 端口被旧服务占用
4. Qwen API 网络或密钥问题
```

### 端口被占用

换一个端口启动：

```cmd
python -B web_app.py --host 127.0.0.1 --port 7872
```

然后浏览器访问：

```text
http://127.0.0.1:7872
```

### 乱码问题

所有网页文件请保持 UTF-8 编码保存，尤其是：

```text
web/index.html
web/app.js
web/styles.css
```

VS Code 右下角可以查看和切换文件编码。
