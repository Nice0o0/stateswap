/* stateswap WebUI — DeepSeek-inspired layout, no build step. */
"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  personas: [],
  sessions: [],
  activeId: null,
  sending: false,
  ready: false,
};

/* ---------- helpers ---------- */
async function api(path, opts) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch {}
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json();
}

function setStatus(ok, text) {
  $("server-status").className = ok ? "dot-on" : "dot-off";
  if (text) $("server-status").title = text;
}

/* ---------- theme ---------- */
function applyThemeIcon() {
  const dark = document.documentElement.dataset.theme === "dark";
  $("btn-theme").textContent = dark ? "☀️" : "🌙";
}
$("btn-theme").onclick = () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  localStorage.setItem("stateswap-theme", next);
  applyThemeIcon();
};
applyThemeIcon();

/* ---------- navigation (DeepSeek: 对话是主视图，其余从侧栏进入) ---------- */
function showView(name) {
  document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
  $("view-" + name).classList.add("active");
  if (name === "mix") fillMixSelects();
  if (name === "train") loadTrainDatasets();
}
$("nav-mix").onclick = () => showView("mix");
$("nav-train").onclick = () => showView("train");
for (const link of document.querySelectorAll(".back")) {
  link.onclick = () => showView(link.dataset.nav);
}

/* ---------- mobile sidebar（窄屏抽屉） ---------- */
function setSidebar(open) {
  document.body.classList.toggle("sidebar-open", open);
  $("backdrop").classList.toggle("show", open);
}
$("btn-menu").onclick = () =>
  setSidebar(!document.body.classList.contains("sidebar-open"));
$("backdrop").onclick = () => setSidebar(false);

/* ---------- personas ---------- */
async function loadPersonas() {
  state.personas = (await api("/v1/personas")).personas;
  renderPersonaSelect();
  fillMixSelects();
}

function renderPersonaSelect() {
  const sel = $("persona-select");
  const prev = sel.value;
  sel.innerHTML = state.personas
    .map((p) => `<option value="${p.name}">${p.name}（${p.size_mb} MB）</option>`)
    .join("");
  const preferred = state.personas.find((p) => p.name === prev)?.name
    || state.personas.find((p) => p.name.startsWith("neko-1.5b"))?.name
    || state.personas.find((p) => p.name.startsWith("neko"))?.name
    || state.personas[0]?.name;
  if (preferred) sel.value = preferred;
}

/* ---------- sessions ---------- */
async function loadSessions() {
  state.sessions = (await api("/v1/sessions")).sessions;
  renderSessions();
}

function renderSessions() {
  const list = $("session-list");
  list.innerHTML = "";
  if (!state.sessions.length) {
    const empty = document.createElement("div");
    empty.className = "muted";
    empty.style.cssText = "font-size:12px;padding:6px 8px;";
    empty.textContent = "暂无会话";
    list.appendChild(empty);
    return;
  }
  for (const s of state.sessions) {
    const item = document.createElement("div");
    item.className = "session-item" + (s.session_id === state.activeId ? " active" : "");

    const info = document.createElement("div");
    info.className = "s-info";
    const name = document.createElement("div");
    name.className = "s-name";
    name.textContent = s.persona; // 用户输入的人格名，textContent 防 XSS
    const meta = document.createElement("div");
    meta.className = "s-meta";
    meta.textContent = `${s.turns} 轮 · ${s.memory_mb} MB`;
    meta.title = "状态大小由底座结构决定（层数×头数×64×64），与对话内容无关";
    info.append(name, meta);

    const del = document.createElement("button");
    del.className = "s-del";
    del.title = "删除会话";
    del.textContent = "✕";
    del.onclick = (ev) => { ev.stopPropagation(); removeSession(s.session_id); };

    item.append(info, del);
    item.onclick = () => activateSession(s.session_id);
    list.appendChild(item);
  }
}

function updateSessionTag(detail) {
  $("session-tag").textContent =
    `会话 ${detail.session_id} · ${detail.turns} 轮 · ${detail.memory_mb} MB 状态`;
  // 人格可能已被删除（不在下拉里），此时保留当前选择避免静默错位
  const sel = $("persona-select");
  if ([...sel.options].some((o) => o.value === detail.persona)) {
    sel.value = detail.persona;
  }
}

function renderHistory(history) {
  const box = $("chat-messages");
  box.innerHTML = "";
  if (!history.length) showWelcome();
  for (const m of history) {
    if (m.role === "user") addUserBubble(m.content);
    else if (m.role === "assistant") addAssistantShell(m.content, null);
  }
  scrollChat(true);
}

/* 空会话欢迎页：示例 prompt 一键填入输入框 */
function showWelcome() {
  const wrap = document.createElement("div");
  wrap.className = "welcome";
  const h = document.createElement("h2");
  h.textContent = "一个底座，N 个人格";
  const p = document.createElement("p");
  p.textContent = "顶栏选择人格（约 3ms 热切换）。试试翻译人格：选中 zh2en 后直接打中文，不需要任何指令。";
  const chips = document.createElement("div");
  chips.className = "chips";
  for (const text of [
    "早上好呀！今天想吃小鱼干吗？",
    "陪我聊聊天吧，今天有点累",
    "今天的月亮又圆又亮。",
    "给我讲个笑话吧",
  ]) {
    const c = document.createElement("button");
    c.className = "chip";
    c.type = "button";
    c.textContent = text;
    c.onclick = () => {
      const el = $("input");
      el.value = text;
      el.dispatchEvent(new Event("input")); // 触发自动增高
      el.focus();
    };
    chips.appendChild(c);
  }
  wrap.append(h, p, chips);
  $("chat-messages").appendChild(wrap);
}

async function activateSession(id) {
  if (state.sending) return; // 流式进行中切会话会孤立正在渲染的气泡
  try {
    const detail = await api(`/v1/sessions/${id}`);
    state.activeId = detail.session_id;
    updateSessionTag(detail);
    renderHistory(detail.history);
    renderSessions();
    setSidebar(false); // 窄屏抽屉：选中后自动收起
  } catch (e) {
    showErr(e);
  }
}

async function newSession() {
  const persona = $("persona-select").value || "none";
  const s = await api("/v1/sessions", { method: "POST", body: JSON.stringify({ persona }) });
  await loadSessions();
  await activateSession(s.session_id);
}

$("btn-new-session").onclick = () => newSession().catch(showErr);

async function removeSession(id) {
  await api(`/v1/sessions/${id}`, { method: "DELETE" });
  await loadSessions();
  if (state.activeId === id) {
    if (state.sessions.length) await activateSession(state.sessions[0].session_id);
    else { state.activeId = null; $("chat-messages").innerHTML = ""; $("session-tag").textContent = ""; }
  }
}

/* ---------- persona switch (hot swap, ~3ms) ---------- */
$("persona-select").onchange = async () => {
  const persona = $("persona-select").value;
  if (!state.activeId) { await newSession().catch(showErr); return; }
  try {
    const r = await api(`/v1/sessions/${state.activeId}/swap`, {
      method: "POST",
      body: JSON.stringify({ persona, keep_context: false }),
    });
    await loadSessions();
    await activateSession(state.activeId);
    addSystemNote(`已切换人格 → ${persona}（${Number(r.latency_ms).toFixed(1)} ms），上下文已重置`);
  } catch (e) {
    showErr(e);
  }
};

/* ---------- chat rendering ---------- */
// 用户上翻阅读历史时不强制回底；自己发消息/切会话时始终回底
let stickToBottom = true;
$("chat-scroll").addEventListener("scroll", () => {
  const sc = $("chat-scroll");
  stickToBottom = sc.scrollHeight - sc.scrollTop - sc.clientHeight < 60;
});

function scrollChat(force = false) {
  if (!force && !stickToBottom) return;
  const sc = $("chat-scroll");
  sc.scrollTop = sc.scrollHeight;
}

function addUserBubble(text) {
  const row = document.createElement("div");
  row.className = "msg-row user";
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = "你";
  const bubble = document.createElement("div");
  bubble.className = "msg-user";
  bubble.textContent = text;
  row.append(avatar, bubble);
  $("chat-messages").appendChild(row);
  scrollChat(true); // 自己发的消息始终回底
}

function addAssistantShell(text, stats) {
  const row = document.createElement("div");
  row.className = "msg-row assistant";
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = "S";
  const block = document.createElement("div");
  block.className = "msg-assistant";
  const body = document.createElement("div");
  body.className = "a-text" + (text ? "" : " empty");
  if (text) {
    body.textContent = text;
  } else {
    body.append("思考中");
    for (let i = 0; i < 3; i++) {
      const d = document.createElement("span");
      d.className = "tdot";
      d.textContent = ".";
      body.appendChild(d);
    }
  }
  block.appendChild(body);
  if (stats) block.appendChild(renderStats(stats));
  row.append(avatar, block);
  $("chat-messages").appendChild(row);
  scrollChat();
  return body;
}

function renderStats(stats) {
  const wrap = document.createElement("div");
  wrap.className = "a-stats";
  for (const t of [
    `${stats.completion_tokens} tok`,
    `prefill ${stats.prefill_ms}ms`,
    `${stats.decode_ms_per_token}ms/tok`,
    `状态 ${stats.session_memory_mb}MB`,
  ]) {
    const chip = document.createElement("span");
    chip.textContent = t;
    wrap.appendChild(chip);
  }
  return wrap;
}

function addSystemNote(text) {
  const div = document.createElement("div");
  div.className = "sys-note";
  div.textContent = text;
  $("chat-messages").appendChild(div);
}

function showErr(e) {
  addSystemNote(`⚠ ${e.message || e}`);
}

/* ---------- send (SSE streaming) ---------- */
// Enter 发送的三层兜底：
// 1) keydown（真实浏览器的主路径），isComposing 时忽略（输入法确认候选词）
// 2) 嵌入式/无键盘事件环境：Enter 以 insertLineBreak 的形式插入换行，
//    在 input 事件里识别并转为发送
// 3) Shift+Enter 换行：keydown 里设置抑制标记，避免兜底误发送
let suppressLineBreakSend = false;
$("input").addEventListener("keydown", (ev) => {
  if (ev.key !== "Enter" || ev.isComposing) return;
  suppressLineBreakSend = ev.shiftKey;
  if (!ev.shiftKey) {
    ev.preventDefault();
    $("composer").requestSubmit();
  }
});

// 输入框随内容自动增高（最多 6 行），发送后复位
$("input").addEventListener("input", (ev) => {
  const el = $("input");
  if (ev.inputType === "insertLineBreak") {
    if (suppressLineBreakSend) {
      suppressLineBreakSend = false; // Shift+Enter：保留换行，不发送
    } else {
      el.value = el.value.replace(/\n+$/, "");
      $("composer").requestSubmit();
    }
  }
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 160) + "px";
});

$("composer").onsubmit = async (ev) => {
  ev.preventDefault();
  if (state.sending) return;
  if (!state.ready) { showErr("引擎尚未就绪（首次加载需要预热内核），稍候再试"); return; }
  const text = $("input").value.trim();
  if (!text) return;
  // 同步置位，杜绝会话创建期间的双击双发
  state.sending = true;
  $("btn-send").disabled = true;
  $("btn-send").classList.add("sending");
  const body = addAssistantShell("", null);

  try {
    if (!state.activeId) await newSession();
    const el = $("input");
    el.value = "";
    el.style.height = "auto";
    addUserBubble(text);

    const temp = Number($("temp-preset").value);
    const res = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: $("persona-select").value,
        session_id: state.activeId,
        stream: true,
        temperature: temp,
        top_p: temp >= 1.0 ? 0.9 : 0.8,
        messages: [{ role: "user", content: text }],
      }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "", reply = "", stats = null, errorMsg = null;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const events = buf.split("\n\n");
      buf = events.pop(); // 末尾可能是不完整事件，留给下一次 read 拼齐
      for (const chunk of events) {
        const line = chunk.trim();
        if (!line.startsWith("data: ") || line === "data: [DONE]") continue;
        const payload = JSON.parse(line.slice(6));
        const delta = payload.choices?.[0]?.delta?.content;
        if (delta) {
          reply += delta;
          body.textContent = reply;
          body.classList.remove("empty");
          scrollChat();
        }
        if (payload.stateswap) {
          stats = {
            ...payload.stateswap,
            completion_tokens: payload.usage?.completion_tokens ?? 0,
          };
        }
        if (payload.error) errorMsg = payload.error.message;
      }
    }
    if (errorMsg) {
      addSystemNote(`⚠ ${errorMsg}`);
      body.closest(".msg-row")?.remove(); // 整行移除，不留孤儿头像
    } else if (stats && stats.degenerated) {
      // 退化护栏：回复虽已流式显示，但未写入会话记忆（状态已回滚）
      body.textContent =
        "（该回复检测到复读/乱码退化，已回滚会话状态、未写入记忆。请换个问法或要求更短的回复后重试）";
      body.classList.add("empty");
      loadSessions();
    } else {
      body.textContent = reply || "（空回复）";
      if (stats) blockAppendStats(body, stats);
      loadSessions(); // 刷新侧栏顺序/轮数
    }
  } catch (e) {
    showErr(e);
  } finally {
    state.sending = false;
    $("btn-send").disabled = false;
    $("btn-send").classList.remove("sending");
    $("input").focus();
    scrollChat();
  }
};

function blockAppendStats(body, stats) {
  body.classList.remove("empty");
  body.appendChild(renderStats(stats));
  const s = state.sessions.find((x) => x.session_id === state.activeId);
  if (s) s.turns = stats.turns;
}

/* ---------- persona mixer ---------- */
function fillMixSelects() {
  const names = state.personas.map((p) => p.name);
  const shapeOf = (n) => JSON.stringify(state.personas.find((x) => x.name === n)?.shape);
  const selA = $("mix-a"), selB = $("mix-b");

  const keepA = names.includes(selA.value) ? selA.value
    : (names.includes("neko-1.5b") ? "neko-1.5b" : names[0]);
  selA.innerHTML = names.map((n) => `<option>${n}</option>`).join("");
  selA.value = keepA;

  const compatible = names.filter((n) => shapeOf(n) === shapeOf(selA.value));
  const keepB = compatible.includes(selB.value) ? selB.value
    : (compatible.includes("zh2en-1.5b") ? "zh2en-1.5b" : compatible[0]);
  selB.innerHTML = compatible.map((n) => `<option>${n}</option>`).join("");
  selB.value = keepB;
}

$("mix-a").onchange = fillMixSelects;
$("mix-alpha").oninput = () => {
  $("mix-alpha-val").textContent = Number($("mix-alpha").value).toFixed(2);
};

$("btn-mix").onclick = async () => {
  const a = $("mix-a").value, b = $("mix-b").value, alpha = Number($("mix-alpha").value);
  const name = $("mix-name").value.trim() || `mix-${a}-x-${b}-${alpha.toFixed(2)}`.replace(/\./g, "p");
  const out = $("mix-result");
  out.className = "muted";
  out.textContent = "混合中…";
  try {
    const r = await api("/v1/personas/mix", {
      method: "POST",
      body: JSON.stringify({ a, b, alpha, name }),
    });
    out.className = "ok";
    out.textContent = `已注册人格 ${r.name}（${r.size_mb} MB）——顶栏切换到它即可对话。`
      + " 提醒：混合体可能整体偏向某一个任务模式、回复混乱（见相变实验），属预期行为；"
      + "想分开用两个人格，直接在顶栏切换即可。";
    await loadPersonas();
  } catch (e) {
    out.className = "err";
    out.textContent = e.message;
  }
};

/* ---------- training ---------- */
async function loadTrainDatasets() {
  if ($("train-data").options.length) return;
  const { datasets } = await api("/v1/train/datasets");
  $("train-data").innerHTML = datasets.map((d) => `<option>${d}</option>`).join("");
}

let pollTimer = null;
$("btn-train").onclick = async () => {
  const payload = {
    data: $("train-data").value,
    persona_name: $("train-name").value.trim(),
    steps: Number($("train-steps").value),
    lr: Number($("train-lr").value),
    ctx: Number($("train-ctx").value),
  };
  if (!payload.persona_name) { setTrainResult("err", "请先填写人格名"); return; }
  try {
    await api("/v1/train/start", { method: "POST", body: JSON.stringify(payload) });
    setTrainResult("muted", "训练已启动…");
    $("train-progress-wrap").hidden = false;
    $("btn-train").disabled = true;
    pollTimer = setInterval(pollTrain, 700);
  } catch (e) {
    setTrainResult("err", e.message);
  }
};

function setTrainResult(cls, text) {
  $("train-result").className = cls === "muted" ? "muted" : cls;
  $("train-result").textContent = text;
}

async function pollTrain() {
  const s = await api("/v1/train/status");
  if (s.running && s.total) {
    const pct = Math.round((s.step / s.total) * 100);
    $("train-progress-bar").style.width = pct + "%";
    $("train-progress-text").textContent =
      `step ${s.step}/${s.total} · loss ${s.loss ?? "…"} · lr ${s.lr ?? "…"}`;
  } else if (s.running === false && pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
    $("btn-train").disabled = false;
    if (s.error) {
      setTrainResult("err", "训练失败：" + s.error);
    } else {
      $("train-progress-bar").style.width = "100%";
      setTrainResult("ok", `完成！人格 ${s.persona} 已自动注册，顶栏切换到它即可对话。`);
      await loadPersonas();
    }
  }
}

/* ---------- boot ---------- */
(async () => {
  // 引擎加载（权重 + 内核预热）可能需要几十秒：轮询 /health 直到就绪
  let ready = false;
  for (let i = 0; i < 90; i++) {
    try {
      const h = await fetch("/health").then((r) => r.json());
      if (h.engine_ready) { ready = true; break; }
      setStatus(false, "引擎加载中…");
    } catch {
      setStatus(false, "连接中…");
    }
    await new Promise((r) => setTimeout(r, 1000));
  }
  if (!ready) {
    setStatus(false, "服务不可达");
    addSystemNote("⚠ 无法连接 stateswap 服务，请确认服务器已启动（python -m stateswap.server ...）");
    return;
  }
  try {
    state.ready = true;
    setStatus(true, "服务正常");
    await loadPersonas();
    await loadSessions();
    if (state.sessions.length) {
      await activateSession(state.sessions[0].session_id);
    } else {
      await newSession();
    }
  } catch (e) {
    showErr(e);
  }
})();
