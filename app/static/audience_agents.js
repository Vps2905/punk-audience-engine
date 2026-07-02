const runBtn = document.getElementById("runBtn");
const latestBtn = document.getElementById("latestBtn");
const statusBox = document.getElementById("status");
const modulesBox = document.getElementById("modules");
const rawResponse = document.getElementById("rawResponse");

function setText(id, value) {
  document.getElementById(id).textContent = value ?? "-";
}

function render(data) {
  const summary = data.summary || {};

  setText("rawRows", summary.raw_rows_loaded);
  setText("sessions", summary.deduped_sessions);
  setText("cohorts", summary.safe_cohort_count);
  setText("clusters", summary.cluster_count);
  setText("synthetic", summary.synthetic_rows);
  setText("exportStatus", summary.approval_status);

  modulesBox.innerHTML = "";

  const zips = data.module_zips || [];

  if (!zips.length) {
    modulesBox.innerHTML = "<p>No module ZIPs found yet.</p>";
  }

  zips.forEach((item) => {
    const div = document.createElement("div");
    div.className = "card";

    const title = item.name
      .replace(".zip", "")
      .replaceAll("_", " ");

    div.innerHTML = `
      <div class="module-title">${title}</div>
      <div class="label">${item.exists ? "Ready" : "Missing"}</div>
      ${
        item.exists
          ? `<p><a href="${item.download_url}" target="_blank">Download</a></p>`
          : ""
      }
      <div class="label">${item.path || ""}</div>
    `;

    modulesBox.appendChild(div);
  });

  rawResponse.textContent = JSON.stringify(data, null, 2);
}

async function loadLatest() {
  statusBox.textContent = "Loading latest output...";
  try {
    const res = await fetch("/agents/latest-five-module-demo-output");
    const data = await res.json();

    if (!res.ok) {
      throw new Error(JSON.stringify(data));
    }

    statusBox.textContent = `Loaded latest package: ${data.latest_package}`;
    render(data);
  } catch (err) {
    statusBox.textContent = `Error: ${err.message}`;
  }
}

async function runDemo() {
  runBtn.disabled = true;
  statusBox.textContent = "Running pipeline. This can take some time...";

  try {
    const res = await fetch("/agents/generate-five-module-demo-output", {
      method: "POST",
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(JSON.stringify(data));
    }

    statusBox.textContent = `Completed. Latest package: ${data.latest_package}`;
    render(data);
  } catch (err) {
    statusBox.textContent = `Error: ${err.message}`;
  } finally {
    runBtn.disabled = false;
  }
}

runBtn.addEventListener("click", runDemo);
latestBtn.addEventListener("click", loadLatest);

loadLatest();
