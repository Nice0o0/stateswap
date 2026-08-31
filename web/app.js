/* stateswap WebUI — no build step, no dependencies. */
"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  personas: [],
  selected: null,
  session: null, // {session_id, persona}
  sending: false,
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
  $("status-text").textContent = text;
}

/* ---------- personas & session ---------- */
async function loadPersonas() {
  state.personas = (await api("/v1/personas")).personas;
  if (!state.selected) {
    const preferred = state.personas.find((p) => p.name === "neko-0.4b-v2")
      || state.personas.find((p) => p.name.startsWith("neko"))
      || state.personas[0];
    state.selected = preferred ? preferred.name : null;
  }
  renderPersonas();
  renderMixSelects();
}

function renderPersonas() {
  const list = $("persona-list");
  list.innerHTML = "";
  for (const p of state.personas) {
    const base = p.shape ? (p.shape[0] >= 24 ? "0.4B" : p.shape[0] >= 12 ? "0.1B" : "") : "";
    const item = document.createElement("div");
    item.className = "persona-item" + (p.name === state.selected ? " selected" : "");
    item.innerHTML = `<span class="p-name">${p.name}</span>`
      + `<span class="p-size">${p.size_mb} MB${base ? " · " + base : ""}</span>`;
    item.onclick = () => {
      state.selected = p.name;
      renderPersonas();
      updateSessionCard();
    };
    list.appendChild(item);
  }
}

function updateSessionCard() {
  const el = $("session-info");
  if (!state.session) {
    el.textContent = "尚未创建会话";
    el.classList.add("muted");
    $("btn-swap").disabled = true;
    return;
  }
  el.classList.remove("muted");
  el.innerHTML = `会话 <span class="mono">${state.session.session_id}</span><br>
    人格 <b>${state.session.persona}</b> · <span class="muted">${state.session.memory_mb ?? "?"} MB 状态</span>`;
  $("btn-swap").disabled = false;
}

async function newSession(persona) {
  const want = persona || state.selected || "none";
  const s = await api("/v1/sessions", { method: "POST", body: JSON.stringify({ persona: want }) });
  state.session = s;
  updateSessionCard();
  $("chat-messages").innerHTML = "";
  addSystemNote(`已创建会话 ${s.session_id}（人格 ${s.persona}，状态 ${s.memory_mb} MB）`);
  return s;
}

$("btn-new-session").onclick = () => newSession(state.session?.persona || state.selected).catch(showErr);
$("btn-swap").onclick = async () => {
  if (!state.session || !state.selected) return;
  try {
    const r = await api(`/v1/sessions/${state.session.session_id}/swap`, {
      method: "POST",
      body: JSON.stringify({ persona: state.selected }),
    });
    state.session.persona = state.selected;
    updateSessionCard();
    $("chat-messages").innerHTML = "";
    addSystemNote(`已切换人格 → ${r.swapped_to}（${r.latency_ms.toFixed(1)} ms），上下文已重置`);
  } catch (e) { showErr(e); }
};

/* ---------- chat (SSE streaming) ---------- */
function addBubble(role, text) {
  const div = document.createElement("div");
  div.className = "msg " + role;
  div.textContent = text;
  $("chat-messages").appendChild(div);
  scrollChat();
  return div;
}
function addSystemNote(text) {
  const div = document.createElement("div");
  div.className = "sys-note";
  div.innerHTML = text;
  $("chat-messages").appendChild(div);
  scrollChat();
}
function scrollChat() {
  const sc = $("chat-scroll");
  sc.scrollTop = sc.scrollHeight;
}
function showErr(e) {
  addSystemNote(`⚠ ${e.message || e}`);
}

$("composer").onsubmit = async (ev) => {
  ev.preventDefault();
  if (state.sending) return;
  const text = $("input").value.trim();
  if (!text) return;
  if (!state.session) {
    try { await newSession(state.selected); } catch (e) { showErr(e); return; }
  }
  $("input").value = "";
  addBubble("user", text);
  const bubble = addBubble("assistant", "…");
  state.sending = true;
  $("btn-send").disabled = true;

  try {
    const res = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: state.session.persona,
        session_id: state.session.session_id,
        stream: true,
        messages: [{ role: "user", content: text }],
      }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "", reply = "", stats = null;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      for (const chunk of buf.split("\n\n")) {
        buf = buf.includes(chunk) ? buf.slice(buf.indexOf(chunk) + chunk.length + 2) : "";
        const line = chunk.trim();
        if (!line.startsWith("data: ") || line === "data: [DONE]") continue;
        const payload = JSON.parse(line.slice(6));
        const delta = payload.choices?.[0]?.delta?.content;
        if (delta) { reply += delta; bubble.textContent = reply; scrollChat(); }
        if (payload.reply !== undefined) stats = payload; // 结束帧（含统计）
      }
    }
    bubble.textContent = reply || "（空回复）";
    if (stats) {
      const s = document.createElement("span");
      s.className = "stats";
      s.textContent = `${stats.completion_tokens} tok · prefill ${stats.prefill_ms}ms · `
        + `${stats.decode_ms_per_token}ms/tok · 会话状态 ${stats.session_memory_mb}MB`;
      bubble.appendChild(s);
      state.session.turns = stats.turns;
      updateSessionCard();
    }
  } catch (e) {
    bubble.remove();
    showErr(e);
  } finally {
    state.sending = false;
    $("btn-send").disabled = false;
    scrollChat();
  }
};

/* ---------- tabs ---------- */
for (const tab of document.querySelectorAll(".tab")) {
  tab.onclick = () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
    tab.classList.add("active");
    $("view-" + tab.dataset.view).classList.add("active");
    if (tab.dataset.view === "mix") renderMixSelects();
    if (tab.dataset.view === "train") loadTrainDatasets();
  };
}

/* ---------- persona mixer ---------- */
function shapeOf(name) {
  const p = state.personas.find((x) => x.name === name);
  return p ? JSON.stringify(p.shape) : null;
}

function fillSelect(sel, names) {
  sel.innerHTML = names.map((n) => `<option>${n}</option>`).join("");
}

function renderMixSelects() {
  const names = state.personas.map((p) => p.name);
  const selA = $("mix-a"), selB = $("mix-b");

  const keepA = names.includes(selA.value) ? selA.value
    : (names.includes("neko-0.4b-v2") ? "neko-0.4b-v2" : names[0]);
  fillSelect(selA, names);
  selA.value = keepA;

  // B 只允许与 A 同形状（同底座）的人格
  const compatible = names.filter((n) => shapeOf(n) === shapeOf(selA.value));
  const keepB = compatible.includes(selB.value) ? selB.value
    : (compatible.includes("zh2en-0.4b-v3") ? "zh2en-0.4b-v3" : compatible[0]);
  fillSelect(selB, compatible);
  selB.value = keepB;
}

$("mix-a").onchange = renderMixSelects;

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
    out.textContent = `已注册人格 ${r.name}（${r.size_mb} MB）——去左侧选中它、开新会话试试行为。`;
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
      setTrainResult("ok", `完成！人格 ${s.persona} 已自动注册，去左侧选中它开聊。`);
      await loadPersonas();
    }
  }
}

/* ---------- boot ---------- */
(async () => {
  try {
    await loadPersonas();
    setStatus(true, "服务正常");
    await newSession(state.selected);
  } catch (e) {
    setStatus(false, "服务不可达");
    addSystemNote("⚠ 无法连接 stateswap 服务，请确认服务器已启动（python -m stateswap.server ...）");
  }
})();
