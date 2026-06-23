# 智盾·聊天诈骗识别系统

本项目是一个面向网络聊天场景的诈骗文本识别系统，基于 ChiFraud baseline 数据与本项目生成的对话数据完成训练和网页集成。系统包含本地 BERT、ChineseBERT 推理，以及可选的 Qwen API 云端大模型对比。

当前交付包主要给前端同学继续开发页面与联调接口使用，不需要重新训练模型。

## 1. 快速启动

解压项目后进入项目根目录：

```cmd
cd /d 解压后的目录
```

如果已经有项目使用的 `nlp` 环境，可以直接运行：

```cmd
start_app.cmd
```

或手动指定 Python：

```cmd
D:\anaconda\envs\nlp\python.exe -B web_app.py --host 127.0.0.1 --port 7871
```

启动成功后访问：

```text
http://127.0.0.1:7871
```

局域网演示可运行：

```cmd
start_app_lan.cmd
```

然后用 `ipconfig` 查看本机 IPv4 地址，其他同一局域网设备访问：

```text
http://本机IPv4地址:7871
```

## 2. 环境配置

推荐使用 Anaconda：

```cmd
conda create -n nlp python=3.10 -y
conda activate nlp
python -m pip install -r requirements.txt
```

如果使用已有环境：

```cmd
D:\anaconda\envs\nlp\python.exe -m pip install -r requirements.txt
```

检查 PyTorch 是否可用：

```cmd
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

`torch.cuda.is_available()` 为 `True` 时会优先使用 GPU；为 `False` 时也可以 CPU 推理，但首次加载和预测会慢一些。

## 3. Qwen API 配置

BERT 和 ChineseBERT 是本地离线模型，不需要网络和 API Key。Qwen API 是云端千问接口，需要配置环境变量。

当前代码不会在源码里写死 API Key。需要使用 Qwen API 时，在启动服务前执行：

```cmd
set DASHSCOPE_API_KEY=你的千问APIKey
```

然后再启动：

```cmd
python -B web_app.py --host 127.0.0.1 --port 7871
```

如果没有配置 API Key，网页里的 BERT 和 ChineseBERT 仍可正常使用，只有选择 Qwen API 时会提示密钥未配置。

## 4. 前端开发入口

前端主要修改：

```text
web/index.html
web/styles.css
web/app.js
web/login.html
```

当前页面基于原始 `fraud.html` 风格重做，并已经接入后端接口。前端同学通常只需要改 `web/` 目录，不需要动模型代码。

新增用户登录系统：本地部署时当前页面会先检查登录状态，未登录会跳转到 `/login`。登录页支持用户名/密码登录和注册新账号，用户的检测历史会保存到后端数据库，数据按用户隔离。

后端服务入口：

```text
web_app.py
```

模型推理入口：

```text
inference_backend.py
```

## 5. 后端接口

### 用户登录与个人历史

```http
POST /api/login
POST /api/register
POST /api/logout
GET /api/me
GET /api/history
DELETE /api/history
```

登录后，用户的检测记录会保存到后端数据库 `data/app.db`，不同用户的数据互相隔离。

### 健康检查

```http
GET /api/health
```

### 获取模型信息

```http
GET /api/models
```

### 聊天诈骗检测

```http
POST /api/predict
Content-Type: application/json
```

注意：当前部署已启用用户登录验证，调用该接口前需要先登录。

请求示例，按多轮对话传入：

```json
{
  "model": "chinesebert",
  "messages": [
    {"role": "用户", "text": "你好，我在网上看到一个兼职刷单，说先垫付 300 元。"},
    {"role": "对方", "text": "名额有限，你现在转账就能进群，晚了就没有资格了。"}
  ]
}
```

也可以直接传一整段文本：

```json
{
  "model": "chinesebert",
  "text": "用户：你好，我在网上看到一个兼职刷单，说先垫付 300 元。\n对方：名额有限，你现在转账就能进群。"
}
```

可选模型：

```text
bert
chinesebert
qwen_api
```

返回结果核心字段：

```json
{
  "ok": true,
  "result": {
    "model": "chinesebert",
    "label": "fraud",
    "label_text": "诈骗",
    "risk_score": 0.98,
    "probabilities": {
      "normal": 0.02,
      "fraud": 0.98
    },
    "elapsed_ms": 120
  }
}
```

说明：BERT 和 ChineseBERT 的概率来自分类 logits 的 softmax；Qwen API 是生成式判断，网页展示的是生成标签，不等价于真实置信度。

## 6. 模型结果

当前网页右下角图表使用同一测试集 `dataset/dialogue_binary_matched_2x/test.tsv`，共 120 条样本，正常 90 条、诈骗 30 条。

| 模型 | Acc | 诈骗类 Precision | 诈骗类 Recall | 诈骗类 F1 |
| --- | ---: | ---: | ---: | ---: |
| BERT | 91.67% | 81.25% | 86.67% | 83.87% |
| ChineseBERT | 95.00% | 90.00% | 90.00% | 90.00% |
| Qwen API | 87.50% | 67.44% | 96.67% | 79.45% |

结论：ChineseBERT 综合表现最好，适合作为默认模型；BERT 速度快，适合作为轻量备用；Qwen API 诈骗召回高，但误报较多，适合做大模型对比展示。

## 7. 主要目录

```text
ChiFraud-main/
├─ web/                         # 前端页面
├─ web_app.py                   # HTTP 服务入口
├─ inference_backend.py         # 统一推理入口
├─ models/                      # BERT / ChineseBERT 模型结构
├─ saved_dict/                  # 已训练权重
├─ pretrained/                  # 本地预训练模型文件
├─ dataset/                     # 训练、验证、测试数据
├─ result_qwen/                 # Qwen API 评估结果
├─ requirements.txt             # Python 依赖
├─ start_app.cmd                # 本机启动脚本
└─ start_app_lan.cmd            # 局域网启动脚本
```

## 8. 常见问题

### 页面能打开，但点击检测失败

先查看启动后端的终端输出。常见原因包括依赖未安装、模型文件缺失、端口被占用、Qwen API Key 未配置或网络不可用。

### 第一次检测很慢

模型是懒加载的，第一次选择某个模型时会加载权重，后续预测会快很多。

### Qwen API 显示密钥未配置

启动前设置：

```cmd
set DASHSCOPE_API_KEY=你的千问APIKey
```

然后重新运行 `start_app.cmd` 或手动启动 `web_app.py`。

### 其他电脑访问不了

使用 `start_app_lan.cmd` 启动，并确认两台电脑在同一局域网内。必要时允许 Windows 防火墙放行 Python 或 7871 端口。

### 文件乱码

所有网页和 Markdown 文件请用 UTF-8 保存。VS Code 右下角可以查看和切换编码。

## 截图识别 OCR 说明

新版工作台支持在“智能检测”页面上传聊天截图并识别文字。该功能依赖可选 OCR 环境：

```cmd
D:\anaconda\envs\nlp\python.exe -m pip install Pillow pytesseract
```

此外还需要安装 Tesseract OCR 主程序，并安装中文识别包 `chi_sim`。如果未配置 OCR，文本粘贴、TXT 上传和模型检测仍可正常使用，页面会在点击“识别截图文字”时提示缺少 OCR 组件。
