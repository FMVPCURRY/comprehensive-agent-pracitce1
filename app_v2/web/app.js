const state = {
  model: "chinesebert",
  models: {},
  messages: [
    { role: "用户", text: "你好，我在网上看到一个兼职刷单，说先垫付 300 元，完成后返还本金和佣金。" },
    { role: "对方", text: "名额有限，你现在转账就能进群，晚了就没有资格了。" }
  ],
  history: []
};

const els = {
  serviceStatus: document.querySelector("#serviceStatus"),
  modelSwitch: document.querySelector("#modelSwitch"),
  messageList: document.querySelector("#messageList"),
  bulkText: document.querySelector("#bulkText"),
  fileInput: document.querySelector("#fileInput"),
  fileHint: document.querySelector("#fileHint"),
  importTextBtn: document.querySelector("#importTextBtn"),
  addMessageBtn: document.querySelector("#addMessageBtn"),
  clearBtn: document.querySelector("#clearBtn"),
  detectBtn: document.querySelector("#detectBtn"),
  sampleBtn: document.querySelector("#sampleBtn"),
  riskMeter: document.querySelector("#riskMeter"),
  riskScore: document.querySelector("#riskScore"),
  meterValue: document.querySelector("#meterValue"),
  riskLabel: document.querySelector("#riskLabel"),
  riskCopy: document.querySelector("#riskCopy"),
  normalProb: document.querySelector("#normalProb"),
  fraudProb: document.querySelector("#fraudProb"),
  normalBar: document.querySelector("#normalBar"),
  fraudBar: document.querySelector("#fraudBar"),
  probabilityNote: document.querySelector("#probabilityNote"),
  modelName: document.querySelector("#modelName"),
  modelNote: document.querySelector("#modelNote"),
  elapsed: document.querySelector("#elapsed"),
  inputChars: document.querySelector("#inputChars"),
  rawOutputBlock: document.querySelector("#rawOutputBlock"),
  rawOutput: document.querySelector("#rawOutput"),
  historyList: document.querySelector("#historyList"),
  clearHistoryBtn: document.querySelector("#clearHistoryBtn")
};

function percent(value) {
  return `${Math.round(Number(value || 0) * 100)}%`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderMessages() {
  els.messageList.innerHTML = state.messages.map((message, index) => `
    <div class="message-row">
      <input value="${escapeHtml(message.role)}" data-field="role" data-index="${index}" aria-label="角色">
      <textarea data-field="text" data-index="${index}" aria-label="消息内容">${escapeHtml(message.text)}</textarea>
      <button class="remove-button" data-remove="${index}" type="button" aria-label="删除">×</button>
    </div>
  `).join("");
}

function renderModelMeta() {
  const model = state.models[state.model];
  els.modelName.textContent = model?.name || state.model;
  els.modelNote.textContent = model?.description || "本地模型";
}

function renderHistory() {
  if (!state.history.length) {
    els.historyList.className = "history-list empty";
    els.historyList.textContent = "还没有检测记录";
    return;
  }

  els.historyList.className = "history-list";
  els.historyList.innerHTML = state.history.slice(0, 5).map(item => `
    <article class="history-item">
      <strong>
        <span>${escapeHtml(item.label)}</span>
        <span>${escapeHtml(item.score)}</span>
      </strong>
      <p>${escapeHtml(item.preview)}</p>
    </article>
  `).join("");
}

function setLoading(isLoading) {
  els.detectBtn.disabled = isLoading;
  els.detectBtn.textContent = isLoading ? "模型推理中..." : "开始检测";
  if (isLoading) {
    els.riskLabel.textContent = "正在检测";
    els.riskCopy.textContent = state.model === "qwen_api"
      ? "正在调用云端 Qwen API，请等待接口返回。"
      : "模型正在分析输入文本，请稍等。";
  }
}

function clearResultVisuals() {
  els.riskMeter.style.setProperty("--risk-angle", "0deg");
  els.riskMeter.style.setProperty("--risk-color", "#3b82f6");
  els.riskScore.textContent = "--";
  if (els.meterValue) els.meterValue.textContent = "--";
  els.normalProb.textContent = "--";
  els.fraudProb.textContent = "--";
  els.normalBar.style.width = "0%";
  els.fraudBar.style.width = "0%";
  els.elapsed.textContent = "--";
  els.inputChars.textContent = "--";
  els.rawOutputBlock.hidden = true;
  els.rawOutput.textContent = "";
}

function importBulkText(text) {
  const normalized = text.replace(/\r\n/g, "\n").trim();
  if (!normalized) return;
  state.messages = [{ role: "待检测文本", text: normalized }];
  renderMessages();
}

function updateResult(result) {
  const fraudScore = Number(result.risk_score || 0);
  const normalScore = Number(result.probabilities?.normal || 0);
  const isFraud = result.label === "fraud";
  const isQwen = result.model === "qwen_lora" || result.model === "qwen_api";

  els.riskMeter.style.setProperty("--risk-angle", `${fraudScore * 360}deg`);
  els.riskMeter.style.setProperty("--risk-color", isFraud ? "#ef4444" : "#3b82f6");
  els.riskScore.textContent = isQwen ? result.label_text : percent(fraudScore);
  if (els.meterValue) els.meterValue.textContent = isQwen ? result.label_text : percent(fraudScore);
  els.riskLabel.textContent = isFraud ? "疑似诈骗" : "暂未发现诈骗";
  els.riskCopy.textContent = isFraud
    ? "模型认为这段文本包含诈骗或违法诱导特征，建议重点关注转账、垫付、验证码、私下联系方式等风险点。"
    : "模型当前判断为正常。真实业务中建议对边界样本继续结合规则库或人工复核。";

  els.normalProb.textContent = isQwen ? "生成标签" : percent(normalScore);
  els.fraudProb.textContent = isQwen ? "生成标签" : percent(fraudScore);
  els.normalBar.style.width = isQwen ? (isFraud ? "0%" : "100%") : percent(normalScore);
  els.fraudBar.style.width = isQwen ? (isFraud ? "100%" : "0%") : percent(fraudScore);
  els.probabilityNote.textContent = isQwen
    ? "Qwen 当前是生成式判断：这里只表示生成标签，不代表真实置信度。"
    : "BERT / ChineseBERT 的概率来自分类 logits 的 softmax。";
  els.modelName.textContent = result.model_name;
  els.elapsed.textContent = `${result.elapsed_ms} ms`;
  els.inputChars.textContent = `${result.input_chars} 字`;

  const rawText = result.raw?.raw_response || "";
  els.rawOutputBlock.hidden = !isQwen;
  els.rawOutput.textContent = rawText || "无原始输出";

  const preview = state.messages.map(message => message.text).join(" ").slice(0, 72);
  state.history.unshift({
    label: result.label_text,
    score: isQwen ? "Qwen 生成标签" : `风险 ${percent(fraudScore)}`,
    preview
  });
  renderHistory();
}

async function loadMeta() {
  try {
    const response = await fetch("/api/models");
    const data = await response.json();
    state.models = data.models || {};
    els.serviceStatus.textContent = "本地服务在线";
    els.serviceStatus.className = "team-badge online";
    renderModelMeta();
  } catch (error) {
    els.serviceStatus.textContent = "后端未连接";
    els.serviceStatus.className = "team-badge offline";
  }
}

async function runDetection() {
  const messages = state.messages
    .map(message => ({ role: message.role.trim(), text: message.text.trim() }))
    .filter(message => message.text);

  if (!messages.length) {
    els.riskLabel.textContent = "请输入内容";
    els.riskCopy.textContent = "至少保留一条有文本的聊天消息，模型才能进行检测。";
    return;
  }

  setLoading(true);
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: state.model, messages })
    });
    const data = await response.json();
    if (!data.ok) throw new Error(data.error || "检测失败");
    updateResult(data.result);
  } catch (error) {
    clearResultVisuals();
    els.riskLabel.textContent = "检测失败";
    els.riskCopy.textContent = error.message;
  } finally {
    setLoading(false);
  }
}

els.messageList.addEventListener("input", event => {
  const index = Number(event.target.dataset.index);
  const field = event.target.dataset.field;
  if (Number.isInteger(index) && field) {
    state.messages[index][field] = event.target.value;
  }
});

els.messageList.addEventListener("click", event => {
  const index = Number(event.target.dataset.remove);
  if (Number.isInteger(index)) {
    state.messages.splice(index, 1);
    if (!state.messages.length) state.messages.push({ role: "用户", text: "" });
    renderMessages();
  }
});

els.modelSwitch.addEventListener("click", event => {
  const button = event.target.closest("[data-model]");
  if (!button) return;
  state.model = button.dataset.model;
  document.querySelectorAll(".model-option").forEach(item => item.classList.toggle("active", item === button));
  renderModelMeta();
});

els.importTextBtn.addEventListener("click", () => importBulkText(els.bulkText.value));
els.fileInput.addEventListener("change", async event => {
  const file = event.target.files?.[0];
  if (!file) return;
  try {
    const text = await file.text();
    els.bulkText.value = text;
    importBulkText(text);
    els.fileHint.textContent = `已导入：${file.name}，${text.length} 字符`;
  } catch (error) {
    els.fileHint.textContent = `文件读取失败：${error.message}`;
  }
});

els.addMessageBtn.addEventListener("click", () => {
  state.messages.push({ role: state.messages.length % 2 ? "用户" : "对方", text: "" });
  renderMessages();
});

els.clearBtn.addEventListener("click", () => {
  state.messages = [{ role: "用户", text: "" }];
  els.bulkText.value = "";
  renderMessages();
});

els.sampleBtn.addEventListener("click", () => {
  state.messages = [
    { role: "陌生客服", text: "你的账户存在风险，需要马上配合核验，否则银行卡会被冻结。" },
    { role: "用户", text: "需要怎么处理？" },
    { role: "陌生客服", text: "先把验证码告诉我，再转 2000 元到安全账户，验证完成会原路退回。" }
  ];
  renderMessages();
});

els.detectBtn.addEventListener("click", runDetection);
els.clearHistoryBtn.addEventListener("click", () => {
  state.history = [];
  renderHistory();
});

renderMessages();
renderHistory();
loadMeta();
