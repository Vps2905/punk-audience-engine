"use strict";

const elements = {
  form: document.getElementById("promptForm"),
  input: document.getElementById("promptInput"),
  send: document.getElementById("sendButton"),
  hero: document.getElementById("heroSection"),
  conversation: document.getElementById("conversation"),
  submittedPrompt: document.getElementById("submittedPrompt"),
  assistantLead: document.getElementById("assistantLead"),
  elapsed: document.getElementById("elapsedTime"),
  result: document.getElementById("resultArea"),
  error: document.getElementById("errorCard"),
  errorMessage: document.getElementById("errorMessage"),
  retry: document.getElementById("retryButton"),
  scroll: document.getElementById("workspaceScroll"),
  recent: document.getElementById("recentRuns"),
  mode: document.getElementById("modeBadge"),
  modeText: document.getElementById("modeText"),
};

const stageElements = Array.from(document.querySelectorAll(".stage"));
const historyKey = "punk-audience-workspace-history-v1";
let timer = null;
let startedAt = 0;
let lastPrompt = "";

function setText(id, value, fallback = "—") {
  const node = document.getElementById(id);
  if (!node) return;
  const normalized = value === null || value === undefined || value === ""
    ? fallback
    : String(value);
  node.textContent = normalized;
}

function label(value) {
  return String(value || "unknown")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function resetStages() {
  stageElements.forEach((stage) => {
    stage.classList.remove("running", "complete", "review");
    const description = stage.querySelector("small");
    if (description) description.textContent = "Queued";
  });
}

function setStage(index, state, description) {
  const stage = stageElements[index];
  if (!stage) return;
  stage.classList.remove("running", "complete", "review");
  stage.classList.add(state);
  const detail = stage.querySelector("small");
  if (detail) detail.textContent = description;
}

function animateStages() {
  const elapsedSeconds = Math.max(
    0,
    Math.floor((Date.now() - startedAt) / 1000),
  );
  elements.elapsed.textContent = `${elapsedSeconds}s`;
  const activeStage = Math.min(4, Math.floor(elapsedSeconds / 3));
  stageElements.forEach((stage, index) => {
    if (index < activeStage) {
      setStage(index, "complete", "Completed");
    } else if (index === activeStage) {
      setStage(index, "running", "Evaluating…");
    }
  });
}

function startProgress() {
  resetStages();
  startedAt = Date.now();
  animateStages();
  timer = window.setInterval(animateStages, 1000);
}

function stopProgress() {
  if (timer !== null) {
    window.clearInterval(timer);
    timer = null;
  }
  const elapsedSeconds = Math.max(
    0,
    Math.floor((Date.now() - startedAt) / 1000),
  );
  elements.elapsed.textContent = `${elapsedSeconds}s`;
}

function completeStages(data) {
  const backendStages = Array.isArray(data.pipeline_stages)
    ? data.pipeline_stages
    : [];
  if (backendStages.length === stageElements.length) {
    backendStages.forEach((stage, index) => {
      const status = String(stage.status || "skipped").toLowerCase();
      const visualState = status === "completed"
        ? "complete"
        : status === "skipped"
          ? "skipped"
          : "review";
      setStage(
        index,
        visualState,
        String(stage.detail || "Not evaluated"),
      );
    });
    return;
  }
  const v2 = data.v2_autonomous || {};
  const freshness = String(data.freshness_status || "").toLowerCase();
  const selected = Number(data.prompt_selected_cohorts || 0);
  const exportEnabled = Boolean(data.downstream_export_enabled);

  setStage(0, "complete", "Privacy controls applied");
  setStage(
    1,
    v2.status === "failed" ? "review" : "complete",
    v2.status === "failed" ? "Review required" : "Retrieval completed",
  );
  setStage(
    2,
    selected > 0 ? "complete" : "review",
    selected > 0 ? `${selected} selected` : "No safe exact match",
  );
  setStage(
    3,
    freshness === "fresh" ? "complete" : "review",
    freshness === "fresh" ? "Fresh evidence" : "Historical review only",
  );
  setStage(
    4,
    "complete",
    exportEnabled ? "Approval state evaluated" : "Guardrails enforced",
  );
}

function uniqueWarnings(data) {
  const v2 = data.v2_autonomous || {};
  const review = data.v2_swarm_review || {};
  return Array.from(new Set([
    ...(data.coverage_warnings || []),
    ...(review.coverage_warnings || []),
    ...(v2.coverage_warnings || []),
    data.block_export_reason,
  ].filter((item) => typeof item === "string" && item.trim())));
}

function appendSummaryLine(container, rawLine) {
  const line = rawLine.trim();
  if (!line) return;
  let node;
  if (line.startsWith("## ")) {
    node = document.createElement("h3");
    node.textContent = line.slice(3);
  } else if (line.startsWith("# ")) {
    node = document.createElement("h2");
    node.textContent = line.slice(2);
  } else if (line.startsWith("- ")) {
    let list = container.lastElementChild;
    if (!list || list.tagName !== "UL") {
      list = document.createElement("ul");
      container.appendChild(list);
    }
    node = document.createElement("li");
    node.textContent = line.slice(2);
    list.appendChild(node);
    return;
  } else {
    node = document.createElement("p");
    node.textContent = line;
  }
  container.appendChild(node);
}

function renderSummary(summary) {
  const container = document.getElementById("businessSummary");
  container.replaceChildren();
  String(summary || "No business summary was returned.")
    .split("\n")
    .forEach((line) => appendSummaryLine(container, line));
}

function renderWarnings(warnings) {
  const card = document.getElementById("warningCard");
  const list = document.getElementById("warningList");
  list.replaceChildren();
  if (!warnings.length) {
    card.classList.add("hidden");
    return;
  }
  warnings.forEach((warning) => {
    const item = document.createElement("li");
    item.textContent = warning;
    list.appendChild(item);
  });
  card.classList.remove("hidden");
}

function renderResult(data) {
  const v2 = data.v2_autonomous || {};
  const embedding = v2.embedding_manifest || {};
  const freshness = data.source_freshness || v2.data_freshness || {};
  const review = data.v2_swarm_review || {};
  const stale = String(data.freshness_status || "").toLowerCase() !== "fresh";
  const exportEnabled = Boolean(data.downstream_export_enabled);
  const terminalPolicy = Boolean(data.terminal_policy_decision);

  completeStages(data);
  setText("runStatus", label(data.status));
  setText(
    "freshnessStatus",
    terminalPolicy ? "Not evaluated" : label(data.freshness_status),
  );
  setText("cohortCount", data.prompt_selected_cohorts, "0");
  setText(
    "activationStatus",
    exportEnabled ? "Approval evaluated" : "Safely blocked",
  );
  setText(
    "sourceRows",
    terminalPolicy
      ? "Not evaluated"
      : data.source_rows || freshness.source_rows_checked,
  );
  setText(
    "vectorCount",
    terminalPolicy ? "Not evaluated" : embedding.vector_count,
  );
  setText(
    "vectorDimension",
    terminalPolicy ? "Not evaluated" : embedding.vector_dimension,
  );
  setText(
    "rankedMatches",
    terminalPolicy ? "Not evaluated" : v2.ranked_match_count,
  );
  setText("runReference", data.run_id ? `Run ${data.run_id}` : "");
  setText(
    "decisionBadge",
    label(data.approval_status || "review required"),
  );
  setText(
    "exportSafetyText",
    exportEnabled
      ? "Delivery eligibility was evaluated"
      : "Downstream export is blocked",
  );

  if (terminalPolicy) {
    const approval = String(data.approval_status || "");
    if (approval === "blocked_privacy_identifier_request") {
      elements.assistantLead.textContent = "The request was blocked by the privacy policy before source data, retrieval, or cohort processing was evaluated.";
    } else if (approval === "blocked_approval_bypass_attempt") {
      elements.assistantLead.textContent = "The request attempted to bypass authoritative safeguards. It was stopped before source access, ranking, activation, or export.";
    } else {
      elements.assistantLead.textContent = "The delivery action was stopped before source access because no explicit approved audience context was supplied.";
    }
    elements.mode.classList.add("stale");
    elements.modeText.textContent = "Policy decision";
  } else {
    elements.assistantLead.textContent = stale
      ? "The analysis completed against historical evidence. The strategy is reviewable, but stale-data controls prevent activation or delivery."
      : "The analysis completed. Review the evidence and governance decision before taking any delivery action.";
    elements.mode.classList.toggle("stale", stale);
    elements.modeText.textContent = stale
      ? "Historical shadow"
      : "Fresh-data evaluation";
  }

  renderSummary(data.business_summary);
  renderWarnings(uniqueWarnings(data));
  elements.result.classList.remove("hidden");
  elements.error.classList.add("hidden");
  saveRun({
    prompt: lastPrompt,
    runId: String(data.run_id || ""),
    status: String(data.status || "completed"),
    freshness: String(data.freshness_status || "unknown"),
    reviewed: String(review.overall_review_status || ""),
    createdAt: new Date().toISOString(),
  });
}

function safeErrorMessage(statusCode) {
  if (statusCode === 401) {
    return "The secure local session expired. Reload the workspace and try again.";
  }
  if (statusCode === 404) {
    return "The local audience workspace is disabled or unavailable.";
  }
  if (statusCode === 422) {
    return "Please provide a clearer audience or campaign goal.";
  }
  return "The governed pipeline stopped safely. Check the server terminal for the engineering error; no audience was activated or exported.";
}

function renderError(message) {
  stopProgress();
  const active = stageElements.findIndex((stage) => stage.classList.contains("running"));
  if (active >= 0) setStage(active, "review", "Stopped safely");
  elements.errorMessage.textContent = message;
  elements.error.classList.remove("hidden");
  elements.result.classList.add("hidden");
  elements.assistantLead.textContent = "The analysis stopped without performing an activation or export.";
}

function resizeInput() {
  elements.input.style.height = "auto";
  elements.input.style.height = `${Math.min(elements.input.scrollHeight, 140)}px`;
}

function showConversation(prompt) {
  elements.hero.classList.add("hidden");
  elements.conversation.classList.remove("hidden");
  elements.result.classList.add("hidden");
  elements.error.classList.add("hidden");
  elements.submittedPrompt.textContent = prompt;
  elements.assistantLead.textContent = "I’m evaluating the request against privacy, retrieval, cohort, evolution, and governance controls.";
  elements.scroll.scrollTo({ top: 0, behavior: "smooth" });
}

async function runAnalysis(prompt) {
  lastPrompt = prompt;
  showConversation(prompt);
  elements.send.disabled = true;
  startProgress();

  try {
    const response = await fetch(
      "/api/audience-intelligence/prompt/ui/run",
      {
        method: "POST",
        credentials: "same-origin",
        headers: {
          "Content-Type": "application/json",
          "X-Punk-Local-UI": "1",
        },
        body: JSON.stringify({ prompt }),
      },
    );
    if (!response.ok) {
      renderError(safeErrorMessage(response.status));
      return;
    }
    const data = await response.json();
    stopProgress();
    renderResult(data);
  } catch (_error) {
    renderError(
      "The browser could not reach the audience service. Confirm the server is running on this address and try again.",
    );
  } finally {
    elements.send.disabled = false;
  }
}

function loadHistory() {
  try {
    const parsed = JSON.parse(localStorage.getItem(historyKey) || "[]");
    return Array.isArray(parsed) ? parsed.slice(0, 8) : [];
  } catch (_error) {
    return [];
  }
}

function renderHistory() {
  const runs = loadHistory();
  elements.recent.replaceChildren();
  if (!runs.length) {
    const empty = document.createElement("p");
    empty.className = "recent-empty";
    empty.textContent = "Your completed analyses will appear here.";
    elements.recent.appendChild(empty);
    return;
  }
  runs.forEach((run) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "recent-run";
    button.textContent = run.prompt || "Audience analysis";
    button.title = run.prompt || "Audience analysis";
    button.addEventListener("click", () => {
      elements.input.value = run.prompt || "";
      resizeInput();
      elements.input.focus();
    });
    elements.recent.appendChild(button);
  });
}

function saveRun(run) {
  const runs = loadHistory();
  runs.unshift(run);
  localStorage.setItem(historyKey, JSON.stringify(runs.slice(0, 8)));
  renderHistory();
}

function resetWorkspace() {
  stopProgress();
  resetStages();
  elements.input.value = "";
  elements.input.style.height = "auto";
  elements.hero.classList.remove("hidden");
  elements.conversation.classList.add("hidden");
  elements.result.classList.add("hidden");
  elements.error.classList.add("hidden");
  elements.input.focus();
}

elements.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const prompt = elements.input.value.trim();
  if (prompt.length < 3 || elements.send.disabled) return;
  runAnalysis(prompt);
});

elements.input.addEventListener("input", resizeInput);
elements.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.form.requestSubmit();
  }
});

document.querySelectorAll(".suggestion").forEach((button) => {
  button.addEventListener("click", () => {
    elements.input.value = button.dataset.prompt || "";
    resizeInput();
    elements.input.focus();
  });
});

document.getElementById("newRunButton").addEventListener("click", resetWorkspace);
elements.retry.addEventListener("click", () => {
  if (lastPrompt) runAnalysis(lastPrompt);
});
document.getElementById("historyNav").addEventListener("click", () => {
  elements.input.focus();
});
document.getElementById("evidenceNav").addEventListener("click", () => {
  const target = document.querySelector(".safety-card");
  if (target && !elements.result.classList.contains("hidden")) {
    target.scrollIntoView({ behavior: "smooth", block: "center" });
  } else {
    elements.input.value = "Explain the privacy, approval, and export controls for the current audience data.";
    resizeInput();
    elements.input.focus();
  }
});

renderHistory();
resizeInput();
