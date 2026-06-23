let conversationMessages = [];
let detectionHistory = [];
let currentModelMeta = {};
let screenshotDataUrl = "";
let metricsChart = null;

const modelMetrics = {
  bert: { name: "BERT", acc: 91.67, fraudP: 81.25, fraudR: 86.67, fraudF1: 83.87, note: "速度快、成本低，适合作为默认本地基线。" },
  chinesebert: { name: "ChineseBERT", acc: 95.00, fraudP: 90.00, fraudR: 90.00, fraudF1: 90.00, note: "融合字形与拼音特征，当前综合效果最好。" },
  qwen_api: { name: "Qwen API", acc: 87.50, fraudP: 67.44, fraudR: 96.67, fraudF1: 79.45, note: "云端生成式判断，诈骗召回高但误报更多。" },
};

const scamCases = [
  { icon: "fa-chart-line", title: "刷单返利", pattern: "先给小额返利建立信任，再要求连续垫付、转账进群或完成组合任务。", caseText: "用户看到“点赞返佣”兼职，前两单得到小额返利，随后被要求垫付 300、1000、5000 元做连单，平台显示可提现但客服要求继续缴纳保证金。", keywords: ["垫付", "返佣", "进群", "连单"], links: [["案例网页", "https://www.baidu.com/s?wd=国家反诈中心+刷单返利+案例"], ["视频检索", "https://www.baidu.com/s?wd=刷单返利+诈骗+视频"]] },
  { icon: "fa-headset", title: "冒充客服", pattern: "以退款、取消会员、账户异常为由，引导屏幕共享或索要验证码。", caseText: "对方自称电商客服，称误开通会员每月扣费，需要按步骤操作解除，诱导下载会议软件并打开屏幕共享。", keywords: ["退款", "验证码", "屏幕共享", "取消会员"], links: [["案例网页", "https://www.baidu.com/s?wd=冒充客服+诈骗+案例"], ["视频检索", "https://www.baidu.com/s?wd=冒充客服+诈骗+视频"]] },
  { icon: "fa-coins", title: "虚假投资", pattern: "宣传内幕消息、稳赚不赔、高收益低风险，并要求向私人账户充值。", caseText: "网友推荐“数字资产量化平台”，前期后台收益持续上涨，小额能提现；追加大额本金后平台冻结账户，客服要求缴税解冻。", keywords: ["稳赚", "内幕", "充值", "解冻金"], links: [["案例网页", "https://www.baidu.com/s?wd=虚假投资理财+诈骗+案例"], ["视频检索", "https://www.baidu.com/s?wd=虚假投资理财+诈骗+视频"]] },
  { icon: "fa-credit-card", title: "贷款引流", pattern: "声称无抵押秒到账，但要求先交手续费、刷流水、保证金或解冻金。", caseText: "对方称贷款额度已批，需要先转“认证金”刷流水证明还款能力；转账后又以银行卡号错误为由要求缴纳解冻费。", keywords: ["无抵押", "手续费", "刷流水", "保证金"], links: [["案例网页", "https://www.baidu.com/s?wd=网络贷款+诈骗+案例"], ["视频检索", "https://www.baidu.com/s?wd=网络贷款+诈骗+视频"]] },
  { icon: "fa-user-secret", title: "冒充熟人/领导", pattern: "通过昵称头像伪装熟人，以紧急周转、项目付款等理由要求转账。", caseText: "头像和昵称都像老师或领导，对方称不方便接电话，要求先帮忙垫付一笔费用，并承诺下午归还。", keywords: ["不方便接电话", "垫付", "领导", "紧急"], links: [["案例网页", "https://www.baidu.com/s?wd=冒充领导熟人+诈骗+案例"], ["视频检索", "https://www.baidu.com/s?wd=冒充领导熟人+诈骗+视频"]] },
  { icon: "fa-heart-crack", title: "情感诱导", pattern: "长期聊天建立亲密关系，再引导投资、借钱或代付。", caseText: "对方以恋爱关系取得信任，随后称发现稳赚投资渠道，指导注册平台充值，最后以账户风控为由无法提现。", keywords: ["恋爱", "稳赚", "充值", "无法提现"], links: [["案例网页", "https://www.baidu.com/s?wd=杀猪盘+诈骗+案例"], ["视频检索", "https://www.baidu.com/s?wd=杀猪盘+诈骗+视频"]] },
];

const $ = (id) => document.getElementById(id);
const escapeHtml = (value) => String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");
const modelLabel = (model) => currentModelMeta[String(model || "").toLowerCase()]?.name || modelMetrics[String(model || "").toLowerCase()]?.name || "未知模型";
const activeModel = () => $("modelSelect").value;

function toast(message, ok = false) {
  const el = $("toast");
  el.textContent = message;
  el.style.background = ok ? "#047857" : "#09090b";
  el.style.display = "block";
  clearTimeout(el.timer);
  el.timer = setTimeout(() => { el.style.display = "none"; }, 3200);
}

function showPage(page) {
  document.querySelectorAll(".page").forEach((el) => el.classList.toggle("active", el.id === `page-${page}`));
  document.querySelectorAll(".nav-item").forEach((el) => el.classList.toggle("active", el.dataset.page === page));
  if (page === "history") loadHistory();
  if (page === "models") setTimeout(renderChart, 50);
}

function setModel(model) {
  $("modelSelect").value = model;
  document.querySelectorAll(".model-card").forEach((el) => el.classList.toggle("active", el.dataset.model === model));
  updateModelInfo();
}

function renderMessages() {
  const container = $("messageList");
  if (!conversationMessages.length) {
    container.innerHTML = `<div class="message-empty">暂无对话，请粘贴、上传或添加消息。</div>`;
    return;
  }
  container.innerHTML = conversationMessages.map((message, index) => `
    <div class="message-item">
      <div class="msg-role">${escapeHtml(message.role)}</div>
      <div class="msg-text">${escapeHtml(message.text)}</div>
      <button class="delete-msg" data-index="${index}" aria-label="删除消息"><i class="fas fa-times"></i></button>
    </div>
  `).join("");
  container.querySelectorAll(".delete-msg").forEach((btn) => {
    btn.addEventListener("click", () => {
      conversationMessages.splice(Number(btn.dataset.index), 1);
      renderMessages();
    });
  });
}

function parseBatchText() {
  const text = $("batchText").value.trim();
  if (!text) return toast("请先粘贴聊天记录");
  const lines = text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  conversationMessages = lines.length <= 1
    ? [{ role: "待检测文本", text }]
    : lines.map((line, index) => {
      const matched = line.match(/^([^:：]{1,14})[:：]\s*(.+)$/);
      return matched ? { role: matched[1], text: matched[2] } : { role: index % 2 ? "用户B" : "用户A", text: line };
    });
  renderMessages();
  toast("已导入到待检测对话", true);
}

function renderHistory() {
  const container = $("historyContainer");
  if (!detectionHistory.length) {
    container.innerHTML = `<section class="history-item"><p class="history-summary">暂无检测记录。完成一次检测后会自动保存到这里。</p></section>`;
    return;
  }
  container.innerHTML = detectionHistory.map((item, index) => {
    const isFraud = item.label_text === "诈骗" || item.label === "fraud";
    const time = item.created_ts ? new Date(item.created_ts * 1000).toLocaleString() : "--";
    return `
      <article class="history-item">
        <div class="history-top">
          <div><span class="history-label">${isFraud ? "疑似诈骗" : "暂未发现诈骗"}</span><span class="model-chip"><i class="fas fa-microchip"></i>${modelLabel(item.model)}</span></div>
          <strong style="color:${isFraud ? "var(--red)" : "var(--green)"}">${Math.round(Number(item.risk_score || 0) * 100)}%</strong>
        </div>
        <p class="history-summary">${escapeHtml(item.preview || "")}</p>
        <div class="button-row"><button class="secondary-btn" data-open-history="${index}"><i class="fas fa-file-lines"></i> 查看原始输入</button><span class="label">${time}</span></div>
        <div class="history-raw"><pre>${escapeHtml(item.input_text || item.preview || "无原始输入记录")}</pre></div>
      </article>
    `;
  }).join("");
  container.querySelectorAll("[data-open-history]").forEach((btn) => btn.addEventListener("click", () => btn.closest(".history-item").classList.toggle("open")));
}

async function loadHistory() {
  try {
    const data = await (await fetch("/api/history")).json();
    if (data.ok) {
      detectionHistory = data.history || [];
      renderHistory();
    }
  } catch (error) {
    console.error(error);
  }
}

function renderAlerts() {
  $("alertGrid").innerHTML = scamCases.map((item) => `
    <article class="alert-card">
      <div class="alert-icon"><i class="fas ${item.icon}"></i></div>
      <h3>${item.title}</h3>
      <p>${item.pattern}</p>
      <div class="case-box"><b>具体案例：</b>${item.caseText}</div>
      <div class="keywords">${item.keywords.map((keyword) => `<span>${keyword}</span>`).join("")}</div>
      <div class="link-row">${item.links.map(([label, href]) => `<a href="${href}" target="_blank" rel="noopener"><i class="fas fa-arrow-up-right-from-square"></i> ${label}</a>`).join("")}</div>
    </article>
  `).join("");
}

function renderModelCards() {
  $("modelCards").innerHTML = Object.entries(modelMetrics).map(([key, item]) => `
    <article class="metric-card">
      <span class="label">${key}</span>
      <h3>${item.name}</h3>
      <p>Accuracy：<b>${item.acc.toFixed(2)}%</b></p>
      <p>诈骗类 Precision / Recall / F1：<b>${item.fraudP.toFixed(2)} / ${item.fraudR.toFixed(2)} / ${item.fraudF1.toFixed(2)}</b></p>
      <p>${item.note}</p>
      <button class="secondary-btn success" onclick="setModel('${key}');showPage('detect')"><i class="fas fa-check"></i> 使用该模型</button>
    </article>
  `).join("");
}

function renderChart() {
  const canvas = $("modelMetricsChart");
  if (!canvas || !window.Chart) return;
  if (metricsChart) metricsChart.destroy();
  metricsChart = new Chart(canvas, {
    type: "bar",
    data: {
      labels: ["BERT", "ChineseBERT", "Qwen API"],
      datasets: [
        { label: "Accuracy", data: [91.67, 95.00, 87.50], backgroundColor: "#111827", borderRadius: 10 },
        { label: "Fraud F1", data: [83.87, 90.00, 79.45], backgroundColor: "#047857", borderRadius: 10 },
        { label: "Fraud Recall", data: [86.67, 90.00, 96.67], backgroundColor: "#dc2626", borderRadius: 10 },
      ],
    },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: "bottom" } }, scales: { y: { beginAtZero: true, max: 100 } } },
  });
}

function updateModelInfo() {
  const model = activeModel();
  const meta = currentModelMeta[model] || {};
  $("modelName").textContent = meta.name || modelMetrics[model]?.name || model;
  $("modelNote").textContent = meta.description || modelMetrics[model]?.note || "当前模型";
}

function parseQwenRaw(rawText) {
  if (!rawText) return {};
  try { return JSON.parse(String(rawText).trim()); } catch (_) {}
  const matched = String(rawText).match(/\{[\s\S]*\}/);
  if (matched) {
    try { return JSON.parse(matched[0]); } catch (_) {}
  }
  return { explanation: String(rawText) };
}

function updateResult(result) {
  const fraud = Number(result.probabilities?.fraud ?? result.risk_score ?? 0);
  const normal = Number(result.probabilities?.normal ?? (1 - fraud));
  const isFraud = result.label === "fraud";
  $("resultArea").style.display = "block";
  $("riskTitle").textContent = isFraud ? "疑似诈骗" : "暂未发现诈骗";
  $("riskTitle").style.color = isFraud ? "var(--red)" : "var(--green)";
  $("meterValue").textContent = `${Math.round(fraud * 100)}%`;
  $("riskMeter").style.setProperty("--risk-angle", `${fraud * 360}deg`);
  $("riskMeter").style.setProperty("--risk-color", isFraud ? "var(--red)" : "var(--green)");
  $("normalProb").textContent = `${Math.round(normal * 100)}%`;
  $("fraudProb").textContent = `${Math.round(fraud * 100)}%`;
  $("normalBar").style.width = `${normal * 100}%`;
  $("fraudBar").style.width = `${fraud * 100}%`;
  $("elapsed").textContent = `${result.elapsed_ms} ms`;
  $("inputChars").textContent = `${result.input_chars} 字`;
  $("explanationBox").textContent = result.model === "qwen_api" ? "Qwen API 返回生成式判断，概率为标签化展示，建议结合解释文本复核。" : "BERT / ChineseBERT 的概率来自分类 logits 的 softmax。";
  updateModelInfo();
  const raw = result.raw?.raw_response || "";
  $("rawOutputBlock").style.display = result.model === "qwen_api" ? "grid" : "none";
  $("rawOutput").textContent = raw || "无原始输出";
  const parsed = parseQwenRaw(raw);
  $("qwenStructured").innerHTML = `<div class="qwen-row"><span>判定结论</span>${escapeHtml(parsed.conclusion || result.label_text)}</div><div class="qwen-row"><span>模型标签</span>${escapeHtml(String(parsed.label ?? result.label ?? "-"))}</div><div class="qwen-row"><span>解释原因</span>${escapeHtml(parsed.explanation || "未返回解释")}</div>`;
  loadHistory();
}

async function runDetection() {
  if (!conversationMessages.length) return toast("请先导入待检测内容");
  const btn = $("detectBtn");
  btn.disabled = true;
  btn.innerHTML = `<i class="fas fa-spinner fa-spin"></i> 检测中`;
  try {
    const data = await (await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: activeModel(), messages: conversationMessages }) })).json();
    if (!data.ok) throw new Error(data.error || "检测失败");
    updateResult(data.result);
    toast("检测完成，已写入历史", true);
  } catch (error) {
    toast(error.message || "检测失败");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<i class="fas fa-robot"></i> 开始检测`;
  }
}

async function loadMeta() {
  try {
    const data = await (await fetch("/api/models")).json();
    currentModelMeta = data.models || {};
    $("serviceStatus").innerHTML = `<i class="fas fa-circle-check"></i> 服务在线`;
    updateModelInfo();
  } catch (_) {
    $("serviceStatus").innerHTML = `<i class="fas fa-triangle-exclamation"></i> 后端未连接`;
  }
}

async function loadUserInfo() {
  try {
    const data = await (await fetch("/api/me")).json();
    if (data.ok && data.user) {
      $("authPanel").style.display = "inline-flex";
      const guest = $("guestPanel");
      if (guest) guest.style.display = "none";
      $("currentUserName").textContent = data.user.username;
      return true;
    }
  } catch (_) {}
  const guest = $("guestPanel");
  if (guest) guest.style.display = "inline-flex";
  const auth = $("authPanel");
  if (auth) auth.style.display = "none";
  return false;
}

async function logout() {
  await fetch("/api/logout", { method: "POST" });
  await loadUserInfo();
  toast("\u5df2\u9000\u51fa\u767b\u5f55", true);
}

async function changePassword() {
  try {
    const data = await (await fetch("/api/change-password", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ old_password: $("oldPassword").value, new_password: $("newPassword").value }) })).json();
    if (!data.ok) throw new Error(data.error || "修改失败");
    $("oldPassword").value = "";
    $("newPassword").value = "";
    toast("密码已更新", true);
  } catch (error) {
    toast(error.message || "修改失败");
  }
}

async function recognizeScreenshot() {
  if (!screenshotDataUrl) return toast("请先上传聊天截图");
  const btn = $("ocrBtn");
  btn.disabled = true;
  btn.innerHTML = `<i class="fas fa-spinner fa-spin"></i> 识别中`;
  try {
    const data = await (await fetch("/api/ocr", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ image: screenshotDataUrl }) })).json();
    if (!data.ok) throw new Error(data.error || "截图识别失败");
    $("batchText").value = data.text;
    parseBatchText();
    toast("截图文字已识别并导入", true);
  } catch (error) {
    toast(error.message || "截图识别失败");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<i class="fas fa-eye"></i> 识别截图文字`;
  }
}

function bindEvents() {
  document.querySelectorAll(".nav-item").forEach((btn) => btn.addEventListener("click", () => showPage(btn.dataset.page)));
  document.querySelectorAll("[data-page-link]").forEach((btn) => btn.addEventListener("click", (event) => { event.preventDefault(); showPage(btn.dataset.pageLink); }));
  document.querySelectorAll(".model-card").forEach((btn) => btn.addEventListener("click", () => setModel(btn.dataset.model)));
  $("parseBatchBtn").addEventListener("click", parseBatchText);
  $("detectBtn").addEventListener("click", runDetection);
  $("logoutBtn").addEventListener("click", logout);
  const accountBtn = $("accountSettingsBtn");
  const accountModal = $("accountModal");
  const closeAccountModal = $("closeAccountModal");
  if (accountBtn && accountModal) accountBtn.addEventListener("click", () => accountModal.classList.add("open"));
  if (closeAccountModal && accountModal) closeAccountModal.addEventListener("click", () => accountModal.classList.remove("open"));
  if (accountModal) accountModal.addEventListener("click", (event) => { if (event.target === accountModal) accountModal.classList.remove("open"); });
  $("changePasswordBtn").addEventListener("click", changePassword);
  $("ocrBtn").addEventListener("click", recognizeScreenshot);
  $("addMsgBtn").addEventListener("click", () => {
    const text = $("newMsgText").value.trim();
    if (!text) return toast("请输入对话内容");
    conversationMessages.push({ role: $("roleSelect").value, text });
    $("newMsgText").value = "";
    renderMessages();
  });
  $("resetConversationBtn").addEventListener("click", () => {
    conversationMessages = [];
    $("batchText").value = "";
    $("resultArea").style.display = "none";
    renderMessages();
  });
  $("loadExampleBtn").addEventListener("click", () => {
    conversationMessages = [
      { role: "用户A", text: "你好，我在网上看到一个兼职刷单，说先垫付 300 元，完成后返还本金和佣金。" },
      { role: "用户B", text: "名额有限，你现在转账就能进群，晚了就没有资格了。" },
    ];
    renderMessages();
  });
  $("clearHistoryBtn").addEventListener("click", async () => {
    const data = await (await fetch("/api/history", { method: "DELETE" })).json();
    if (data.ok) {
      detectionHistory = [];
      renderHistory();
      toast("历史已清空", true);
    } else {
      toast(data.error || "清空失败");
    }
  });
  $("fileUpload").addEventListener("change", async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    $("batchText").value = await file.text();
    parseBatchText();
    event.target.value = "";
  });
  $("screenshotUpload").addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      screenshotDataUrl = reader.result;
      const preview = $("screenshotPreview");
      preview.src = screenshotDataUrl;
      preview.style.display = "block";
      toast("截图已加载，可点击识别截图文字", true);
    };
    reader.readAsDataURL(file);
  });
}

async function init() {
  bindEvents();
  await loadUserInfo();
  await loadMeta();
  await loadHistory();
  renderAlerts();
  renderModelCards();
  renderChart();
  conversationMessages = [
    { role: "用户A", text: "你好，我在网上看到一个兼职刷单，说先垫付 300 元，完成后返还本金和佣金。" },
    { role: "用户B", text: "名额有限，你现在转账就能进群，晚了就没有资格了。" },
  ];
  renderMessages();
}

init();
