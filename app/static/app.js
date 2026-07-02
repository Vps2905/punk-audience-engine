const responseBox = document.getElementById("responseBox");
const summaryBox = document.getElementById("summary");
const statusPill = document.getElementById("statusPill");

function setStatus(label, mode) {
  statusPill.textContent = label;
  statusPill.className = `status-pill ${mode}`;
}

function showResponse(data) {
  responseBox.textContent = JSON.stringify(data, null, 2);
}

function showError(error) {
  setStatus("Error", "error");
  const data = {
    status: "error",
    message: error.message || String(error)
  };
  showResponse(data);
  summaryBox.className = "summary";
  summaryBox.innerHTML = `<strong>Error:</strong> ${data.message}`;
}

function traitsToList(traits) {
  if (!traits || Object.keys(traits).length === 0) {
    return "<p>No aggregated traits returned.</p>";
  }

  const items = [];

  for (const [trait, values] of Object.entries(traits)) {
    const topValues = Object.keys(values).join(", ");
    items.push(`<li><strong>${trait}</strong>: ${topValues}</li>`);
  }

  return `<ul>${items.join("")}</ul>`;
}

function renderSummary(data) {
  const quality = data.audience_summary?.quality || {};
  const traits = data.audience_summary?.aggregated_traits || {};
  const exportData = data.meta_export || {};

  summaryBox.className = "summary";
  summaryBox.innerHTML = `
    <div class="summary-grid">
      <div class="summary-card">
        <span>Job ID</span>
        <strong>${data.job_id || "-"}</strong>
      </div>
      <div class="summary-card">
        <span>Cohort ID</span>
        <strong>${data.cohort_id || "-"}</strong>
      </div>
      <div class="summary-card">
        <span>Export ID</span>
        <strong>${data.export_id || "-"}</strong>
      </div>
      <div class="summary-card">
        <span>Quality Score</span>
        <strong>${quality.quality_score ?? "-"} / 100 (${quality.status || "unknown"})</strong>
      </div>
      <div class="summary-card">
        <span>Synthetic Rows Exported</span>
        <strong>${exportData.synthetic_rows_exported ?? "-"}</strong>
      </div>
      <div class="summary-card">
        <span>Approval Status</span>
        <strong>${exportData.approval_status || "-"}</strong>
      </div>
    </div>

    <div class="traits">
      <h3>Aggregated Audience Traits</h3>
      ${traitsToList(traits)}
    </div>

    <div class="traits">
      <h3>Privacy Guarantees</h3>
      <ul>
        <li>No raw email exported</li>
        <li>No raw phone exported</li>
        <li>No device ID exported</li>
        <li>Aggregated traits only</li>
        <li>Synthetic seed profiles only</li>
        <li>Manual approval before upload</li>
      </ul>
    </div>
  `;
}

async function generateAudience() {
  try {
    const fileInput = document.getElementById("csvFile");
    const prompt = document.getElementById("prompt").value.trim();
    const kMin = document.getElementById("kMin").value;
    const epsilon = document.getElementById("epsilon").value;
    const syntheticRows = document.getElementById("syntheticRows").value;
    const seedLimit = document.getElementById("seedLimit").value;

    if (!fileInput.files.length) {
      alert("Please upload a CSV file.");
      return;
    }

    if (!prompt) {
      alert("Please enter an audience request.");
      return;
    }

    setStatus("Running", "running");
    summaryBox.className = "summary";
    summaryBox.innerHTML = "Running privacy-safe audience pipeline...";

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    formData.append("prompt", prompt);
    formData.append("k_min", kMin);
    formData.append("epsilon", epsilon);
    formData.append("synthetic_rows", syntheticRows);
    formData.append("seed_limit", seedLimit);

    const res = await fetch("/audience/generate", {
      method: "POST",
      body: formData
    });

    const data = await res.json();

    if (!res.ok || data.status === "failed") {
      throw new Error(data.message || "Pipeline failed");
    }

    setStatus("Completed", "done");
    renderSummary(data);
    showResponse(data);
  } catch (error) {
    showError(error);
  }
}
