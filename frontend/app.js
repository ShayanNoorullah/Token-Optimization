const STORAGE_KEY = "token_ctx_session";
const API_BASE = "";

const SAMPLE_PROMPTS = [
  "Hi, my name is Alex and I prefer Python for all my projects.",
  "I'm building a token-efficient chatbot for my university thesis.",
  "Which vector database should I use for local deployment?",
  "How do I set up Qdrant with Docker?",
  "Remind me, what language did I say I prefer?",
];

const state = {
  userId: null,
  sessionId: null,
  email: "",
  turns: [],
  loading: false,
};

async function api(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json();
}

function showError(msg) {
  const el = document.getElementById("errorBanner");
  el.textContent = msg;
  el.classList.remove("hidden");
}

function clearError() {
  const el = document.getElementById("errorBanner");
  el.classList.add("hidden");
}

function renderTurnCard(turn, index) {
  const card = document.createElement("div");
  card.className = "turn-card";

  const req = document.createElement("div");
  req.className = "request-section";
  req.innerHTML = `
    <div class="section-label">Request #${index + 1}</div>
    <div class="request-text">${escapeHtml(turn.request)}</div>
    ${turn.stats ? `
      <div class="request-meta">
        <span class="meta-chip">Query tokens: ${turn.stats.context_breakdown.query_tokens}</span>
        <span class="meta-chip">Short-term msgs: ${turn.stats.short_term_message_count}</span>
        <span class="meta-chip">Facts: ${turn.stats.user_facts.length}</span>
        <span class="meta-chip">Retrieved: ${turn.stats.retrieved_memories.length}</span>
        ${turn.stats.has_session_summary ? '<span class="meta-chip">Summary: yes</span>' : ""}
      </div>
      ${turn.stats.assembled_context_preview ? `
        <div class="context-preview">${escapeHtml(turn.stats.assembled_context_preview)}</div>
      ` : ""}
    ` : ""}
  `;

  const resp = document.createElement("div");
  resp.className = "response-section";
  if (turn.response) {
    const s = turn.stats;
    resp.innerHTML = `
      <div class="section-label">Response #${index + 1}</div>
      <div class="response-text">${escapeHtml(turn.response)}</div>
      ${s ? `
        <div class="response-meta">
          <span class="meta-chip">Context: ${s.context_tokens_used} tokens</span>
          <span class="meta-chip">Naive-18: ${s.naive_baseline_tokens}</span>
          <span class="meta-chip">Savings: ${s.savings_percent}%</span>
          <span class="meta-chip">Response: ${s.response_tokens} tokens</span>
          <span class="meta-chip">Latency: ${s.latency_ms} ms</span>
        </div>
      ` : ""}
    `;
  } else {
    resp.innerHTML = `<div class="section-label">Response</div><div class="placeholder">Waiting...</div>`;
  }

  card.appendChild(req);
  card.appendChild(resp);
  return card;
}

function renderTurns() {
  const container = document.getElementById("turnsContainer");
  container.innerHTML = "";
  if (state.turns.length === 0) {
    container.innerHTML = '<p class="placeholder">Send a message to start testing.</p>';
    return;
  }
  state.turns.forEach((turn, i) => container.appendChild(renderTurnCard(turn, i)));
  container.scrollTop = container.scrollHeight;
}

function renderStats(stats) {
  const el = document.getElementById("statsPanel");
  if (!stats) { el.innerHTML = '<p class="placeholder">Stats appear after first response.</p>'; return; }

  const b = stats.context_breakdown;
  const savingsClass = stats.savings_percent >= 0 ? "positive" : "negative";
  const slots = [
    { label: "User Facts", tokens: b.user_facts_tokens, color: "#6366f1" },
    { label: "Session Summary", tokens: b.session_summary_tokens, color: "#8b5cf6" },
    { label: "Retrieved", tokens: b.retrieved_memories_tokens, color: "#a855f7" },
    { label: "Short-term", tokens: b.short_term_tokens, color: "#d946ef" },
    { label: "Query", tokens: b.query_tokens, color: "#ec4899" },
    { label: "Overhead", tokens: b.overhead_tokens, color: "#94a3b8" },
  ];

  el.innerHTML = `
    <div class="stat-grid">
      <div class="stat-card highlight"><span class="stat-label">Context</span><span class="stat-value">${stats.context_tokens_used}</span></div>
      <div class="stat-card"><span class="stat-label">Naive-18</span><span class="stat-value">${stats.naive_baseline_tokens}</span></div>
      <div class="stat-card highlight"><span class="stat-label">Savings</span><span class="stat-value">${stats.savings_percent}%</span></div>
      <div class="stat-card"><span class="stat-label">Latency</span><span class="stat-value">${stats.latency_ms}ms</span></div>
    </div>
    <h3>Context Breakdown</h3>
    <div class="breakdown-bars">
      ${slots.map(s => `
        <div class="breakdown-row">
          <span class="breakdown-label">${s.label}</span>
          <div class="breakdown-bar-bg">
            <div class="breakdown-bar-fill" style="width:${Math.min(100, (s.tokens / b.token_budget) * 100)}%;background:${s.color}"></div>
          </div>
          <span class="breakdown-value">${s.tokens}</span>
        </div>
      `).join("")}
    </div>
    <div class="budget-line">
      Total: <strong>${b.total_context_tokens}</strong> / ${b.token_budget}
      <span class="savings-badge ${savingsClass}">${Math.abs(stats.savings_percent)}% ${stats.savings_percent >= 0 ? "saved" : "over"}</span>
    </div>
  `;
}

function renderChart() {
  const el = document.getElementById("tokenChart");
  const data = state.turns.filter(t => t.stats);
  if (!data.length) { el.innerHTML = '<p class="placeholder">Chart appears after first response.</p>'; return; }

  const maxVal = Math.max(...data.map(t => Math.max(t.stats.context_tokens_used, t.stats.naive_baseline_tokens)), 1);
  el.innerHTML = `
    <div class="chart-bars">
      ${data.map((t, i) => {
        const s = t.stats;
        const ctxW = (s.context_tokens_used / maxVal) * 100;
        const naiveW = (s.naive_baseline_tokens / maxVal) * 100;
        return `
          <div class="chart-row">
            <span class="chart-label">#${i + 1}</span>
            <div class="chart-bar-group">
              <div class="chart-bar context" style="width:${ctxW}%" title="Context: ${s.context_tokens_used}"></div>
              <div class="chart-bar naive" style="width:${naiveW}%" title="Naive-18: ${s.naive_baseline_tokens}"></div>
            </div>
            <span class="chart-legend">${s.savings_percent}%</span>
          </div>
        `;
      }).join("")}
    </div>
    <p class="placeholder" style="margin-top:0.5rem">Purple = context tokens | Red = naive-18 baseline</p>
  `;
}

function renderRetrieval(stats) {
  const el = document.getElementById("retrievalPanel");
  if (!stats || !stats.retrieved_memories.length) {
    el.innerHTML = `<p class="placeholder">No memories above threshold (${stats?.retrieval_threshold ?? 0.72}).</p>`;
    return;
  }
  el.innerHTML = `<div class="memory-list">${stats.retrieved_memories.map(m => `
    <div class="memory-card">
      <div class="memory-header">
        <span class="memory-type">${escapeHtml(m.memory_type)}</span>
        <span class="memory-score ${m.score >= stats.retrieval_threshold ? "pass" : "fail"}">${m.score.toFixed(4)}</span>
      </div>
      <div class="memory-content">${escapeHtml(m.content)}</div>
    </div>
  `).join("")}</div>`;
}

function renderFacts(facts) {
  const el = document.getElementById("factsPanel");
  if (!facts || !facts.length) {
    el.innerHTML = '<p class="placeholder">No facts extracted yet.</p>';
    return;
  }
  el.innerHTML = `<table class="facts-table"><thead><tr><th>Key</th><th>Value</th><th>Conf.</th></tr></thead>
    <tbody>${facts.map(f => `<tr><td class="mono">${escapeHtml(f.fact_key)}</td><td>${escapeHtml(f.fact_value)}</td><td>${(f.confidence * 100).toFixed(0)}%</td></tr>`).join("")}</tbody></table>`;
}

async function renderMetrics() {
  const el = document.getElementById("metricsPanel");
  try {
    const m = await api("/metrics/summary");
    el.innerHTML = `
      <div class="stat-grid compact">
        <div class="stat-card"><span class="stat-label">Requests</span><span class="stat-value">${m.total_requests}</span></div>
        <div class="stat-card"><span class="stat-label">Avg Context</span><span class="stat-value">${m.avg_context_tokens}</span></div>
        <div class="stat-card"><span class="stat-label">Avg Naive</span><span class="stat-value">${m.avg_naive_baseline_tokens}</span></div>
        <div class="stat-card highlight"><span class="stat-label">Avg Savings</span><span class="stat-value">${m.avg_savings_percent}%</span></div>
        <div class="stat-card"><span class="stat-label">Avg Retrieved</span><span class="stat-value">${m.avg_retrieval_count}</span></div>
      </div>`;
  } catch {
    el.innerHTML = '<p class="placeholder">Could not load metrics.</p>';
  }
}

function escapeHtml(text) {
  const d = document.createElement("div");
  d.textContent = text;
  return d.innerHTML;
}

function updateSessionUI() {
  document.getElementById("sessionEmail").textContent = state.email || "Not started";
  document.getElementById("sessionId").textContent = state.sessionId ? `Session: ${state.sessionId.slice(0, 8)}...` : "";
}

async function initSession() {
  clearError();
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored) {
    const data = JSON.parse(stored);
    state.userId = data.userId;
    state.sessionId = data.sessionId;
    state.email = data.email;
    updateSessionUI();
    await renderMetrics();
    return;
  }

  const email = `tester+${Date.now()}@uni.edu`;
  const user = await api("/users", { method: "POST", body: JSON.stringify({ email }) });
  const session = await api("/sessions", { method: "POST", body: JSON.stringify({ user_id: user.id, title: "Testing Session" }) });
  state.userId = user.id;
  state.sessionId = session.id;
  state.email = email;
  localStorage.setItem(STORAGE_KEY, JSON.stringify({ userId: user.id, sessionId: session.id, email }));
  updateSessionUI();
  await renderMetrics();
}

async function newSession() {
  localStorage.removeItem(STORAGE_KEY);
  state.turns = [];
  state.userId = null;
  state.sessionId = null;
  state.email = "";
  renderTurns();
  renderStats(null);
  renderChart();
  renderRetrieval(null);
  renderFacts([]);
  await initSession();
}

async function sendMessage(message) {
  if (!state.userId || !state.sessionId || state.loading) return;
  state.loading = true;
  clearError();

  state.turns.push({ request: message, response: null, stats: null });
  renderTurns();

  try {
    const stats = await api("/chat", {
      method: "POST",
      body: JSON.stringify({ user_id: state.userId, session_id: state.sessionId, message }),
    });
    state.turns[state.turns.length - 1] = {
      request: message,
      response: stats.response,
      stats,
    };
    renderTurns();
    renderStats(stats);
    renderChart();
    renderRetrieval(stats);
    renderFacts(stats.user_facts);
    await renderMetrics();
  } catch (e) {
    state.turns.pop();
    renderTurns();
    showError(e.message);
  } finally {
    state.loading = false;
  }
}

function initSamplePrompts() {
  const container = document.getElementById("samplePrompts");
  SAMPLE_PROMPTS.forEach(p => {
    const btn = document.createElement("button");
    btn.className = "chip";
    btn.type = "button";
    btn.textContent = p.slice(0, 50) + "...";
    btn.title = p;
    btn.addEventListener("click", () => sendMessage(p));
    container.appendChild(btn);
  });
}

document.getElementById("chatForm").addEventListener("submit", (e) => {
  e.preventDefault();
  const input = document.getElementById("messageInput");
  const msg = input.value.trim();
  if (!msg) return;
  input.value = "";
  sendMessage(msg);
});

document.getElementById("newSessionBtn").addEventListener("click", newSession);

initSamplePrompts();
initSession().catch(e => showError(e.message));
