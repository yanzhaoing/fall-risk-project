const levelClass = {
  low: "risk-low",
  medium: "risk-medium",
  high: "risk-high",
  critical: "risk-critical",
};

const state = {
  scenarios: [],
  currentScenario: null,
};

function levelLabel(level) {
  const labels = {
    low: "低风险",
    medium: "中风险",
    high: "高风险",
    critical: "极高风险",
  };
  return labels[level] || level || "--";
}

function setText(id, value) {
  const node = document.getElementById(id);
  if (node) {
    node.textContent = value;
  }
}

function setLevelText(id, level, fallback) {
  const node = document.getElementById(id);
  if (!node) return;
  node.className = levelClass[level] || "";
  node.textContent = fallback;
}

function renderScenarioButtons(items) {
  const wrap = document.getElementById("scenario-strip");
  wrap.innerHTML = "";
  items.forEach((item, idx) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "scenario-btn" + (idx === 0 ? " is-active" : "");
    btn.dataset.scenario = item.id;
    btn.innerHTML = `<strong>${item.name}</strong><span>${item.description}</span>`;
    btn.addEventListener("click", () => runScenario(item.id));
    wrap.appendChild(btn);
  });
}

function activateScenarioButton(scenarioId) {
  document.querySelectorAll(".scenario-btn").forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.scenario === scenarioId);
  });
}

function renderFeatureList(features = {}) {
  const wrap = document.getElementById("feature-list");
  wrap.innerHTML = "";
  Object.entries(features).forEach(([key, value]) => {
    const row = document.createElement("div");
    row.className = "feature-item";
    row.innerHTML = `<span>${key}</span><strong>${value}</strong>`;
    wrap.appendChild(row);
  });
}

function renderScenarioResult(result) {
  state.currentScenario = result.scenario;
  activateScenarioButton(result.scenario);

  document.getElementById("scene-frame").src = result.visual?.frame_image || "";
  setText("score-value", `${result.score}/100`);
  setLevelText("level-value", result.level, result.level_label || levelLabel(result.level));
  setText("action-value", result.response_action || "--");
  setText("scenario-name", result.scenario || "--");
  setText("notify-value", result.should_notify ? "需要通知" : "继续监控");
  setText("behavior-value", `${result.modalities?.behavior ?? "--"}`);
  setText("environment-value", `${result.modalities?.environment ?? "--"}`);
  setText("baseline-value", `${result.modalities?.baseline ?? "--"}`);
  setText("targets-value", (result.notify_targets || []).join(", ") || "无");
  renderFeatureList(result.features || {});
}

function renderVideoSummary(summary = {}) {
  setText("video-source", summary.source_label || "--");
  setText("video-count", `${summary.sample_count ?? "--"}`);
  setText("video-avg", `${summary.avg_score ?? "--"}`);
  setText("video-peak", `${summary.peak_score ?? "--"}`);
  setLevelText("video-level", summary.peak_level, levelLabel(summary.peak_level));
}

function renderVideoFrames(frames = []) {
  const wrap = document.getElementById("video-frames");
  wrap.innerHTML = "";
  frames.forEach((frame) => {
    const item = document.createElement("article");
    item.className = "frame-card";
    item.innerHTML = `
      <img src="${frame.image}" alt="${frame.caption || "frame"}" />
      <strong class="${levelClass[frame.level] || ""}">${frame.caption || "Frame"}</strong>
      <span>风险 ${frame.score} / 100 · ${levelLabel(frame.level)}</span>
    `;
    wrap.appendChild(item);
  });
}

async function fetchJson(url, options = {}) {
  const res = await fetch(url, options);
  if (!res.ok) {
    throw new Error(`request failed: ${res.status}`);
  }
  return await res.json();
}

async function loadScenarios() {
  const data = await fetchJson("/api/demo/scenarios");
  state.scenarios = data.scenarios || [];
  renderScenarioButtons(state.scenarios);
  if (state.scenarios.length) {
    await runScenario(state.scenarios[0].id);
  }
}

async function loadReadiness() {
  const data = await fetchJson("/api/project/readiness");
  const artifacts = data.artifacts || {};
  setText("artifact-demo", artifacts.demo_evaluation ? "已准备" : "待生成");
  setText("artifact-public", artifacts.public_ntu_evaluation ? "已准备" : "待生成");
  setText("artifact-readiness", artifacts.submission_readiness ? "已准备" : "待生成");
  setText("artifact-needed", (data.next_needed || []).length ? `${data.next_needed.length} 项` : "无");
}

async function runScenario(scenario) {
  const data = await fetchJson("/api/demo/evaluate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scenario }),
  });
  renderScenarioResult(data);
}

async function loadSampleVideo() {
  const data = await fetchJson("/api/video/sample");
  renderVideoSummary(data.summary || {});
  renderVideoFrames(data.frames || []);
}

async function uploadVideo() {
  const file = document.getElementById("video-file").files?.[0];
  if (!file) return;
  const form = new FormData();
  form.append("video", file);
  const data = await fetchJson("/api/video/analyze", {
    method: "POST",
    body: form,
  });
  renderVideoSummary(data.summary || {});
  renderVideoFrames(data.frames || []);
}

document.getElementById("sample-video-btn").addEventListener("click", loadSampleVideo);
document.getElementById("upload-video-btn").addEventListener("click", uploadVideo);

loadScenarios().catch((err) => {
  setText("server-state", `服务异常: ${err.message}`);
});

loadReadiness().catch(() => {});
