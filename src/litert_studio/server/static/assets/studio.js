const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const dialog = $("#result-dialog");

function showView(id) {
  $$(".view").forEach(v => v.classList.toggle("active", v.id === id));
  $$("#nav button").forEach(b => b.classList.toggle("active", b.dataset.view === id));
  const titles = {overview:"Model operations, in one place.",models:"Model library",data:"Dataset quality",train:"Fine-tuning",convert:"LiteRT conversion",validate:"Validation gates",deploy:"Deployment quality",artifacts:"Release artifacts",jobs:"Job history"};
  $("#title").textContent = titles[id];
  if (id === "jobs") loadJobs();
  if (id === "deploy") loadDeployment();
}

async function request(endpoint, payload, button) {
  if (button) { button.disabled = true; button.dataset.label = button.textContent; button.textContent = "Working…"; }
  try {
    const response = await fetch(endpoint, {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || `Request failed (${response.status})`);
    showResult(body, "Operation complete");
    await loadJobs();
    return body;
  } catch (error) {
    showResult({error:error.message}, "Needs attention");
  } finally {
    if (button) { button.disabled = false; button.textContent = button.dataset.label; }
  }
}

function showResult(value, title) {
  $("#result-title").textContent = title;
  $("#result").textContent = JSON.stringify(value, null, 2);
  dialog.showModal();
}

function formPayload(form) {
  const payload = {};
  for (const [key, raw] of new FormData(form)) {
    if (raw === "") continue;
    const input = form.elements[key];
    payload[key] = input?.type === "number" ? Number(raw) : raw;
  }
  if (payload.reports) payload.reports = payload.reports.split(",").map(x => x.trim()).filter(Boolean);
  return payload;
}

function trainingPayload(form) {
  const value = formPayload(form);
  value.schema_version = "1"; value.kind = "training";
  value.lora = {rank:value.lora_rank,alpha:value.lora_rank * 2,dropout:0.05,target_modules:"auto"};
  delete value.lora_rank;
  return value;
}

function conversionPayload(form) {
  const value = formPayload(form);
  value.schema_version = "1"; value.kind = "conversion"; value.output_format = "litertlm";
  value.prefill_lengths = String(value.prefill_lengths || "128").split(",").map(Number);
  return value;
}

async function loadStatus() {
  try {
    const [health, caps, jobs, system] = await Promise.all([fetch("/health").then(r=>r.json()),fetch("/api/capabilities").then(r=>r.json()),fetch("/api/jobs").then(r=>r.json()),fetch("/api/system").then(r=>r.json())]);
    $("#health").textContent = "Ready · " + health.workspace;
    $("#family-count").textContent = caps.conversion[0]?.model_families?.length || 0;
    $("#quant-count").textContent = caps.quantization.filter(item => item.support !== "separate_pipeline").length;
    $("#job-count").textContent = jobs.length;
    const device = system.accelerator.devices[0];
    $("#accelerator").textContent = device ? "GPU" : "CPU";
    $("#accelerator-detail").textContent = device ? `${device.name} · ${device.memory_gib} GiB` : `${system.cpu_count || "?"} logical cores`;
  } catch { $("#health").textContent = "Unavailable"; }
}

async function loadJobs() {
  try {
    const jobs = await fetch("/api/jobs").then(r=>r.json());
    $("#job-count").textContent = jobs.length;
    $("#jobs-list").innerHTML = jobs.length ? jobs.map(job => `<div class="job" data-job="${job.job_id}"><strong>${escapeHtml(job.name)}</strong><span class="state">${escapeHtml(job.state)}</span><p>${escapeHtml(job.kind)} · ${new Date(job.updated_at).toLocaleString()}</p><div class="job-tools"><button class="events">Events</button><button class="logs">Live log</button>${["queued","running"].includes(job.state) ? '<button class="cancel">Cancel</button>' : ""}</div></div>`).join("") : '<p class="empty">No jobs yet.</p>';
    $$(".job").forEach(row => {
      $(".events", row).onclick = async () => showResult(await fetch(`/api/jobs/${row.dataset.job}/events`).then(r=>r.json()), "Job events");
      $(".logs", row).onclick = async () => { const data=await fetch(`/api/jobs/${row.dataset.job}/log`).then(r=>r.json()); showResult(data.log || "No log output yet.", "Worker log"); };
      const cancel=$(".cancel", row); if(cancel) cancel.onclick = () => request(`/api/jobs/${row.dataset.job}/cancel`, {}, cancel);
    });
  } catch {}
}

async function loadDeployment() {
  try {
    const [android, results] = await Promise.all([
      fetch("/api/devices/android").then(r => r.json()),
      fetch("/api/compatibility").then(r => r.json())
    ]);
    $("#adb-status").textContent = android.adb_available ? `${android.devices.length} found` : "Not installed";
    $("#adb-status").classList.toggle("good", android.devices.some(device => device.state === "device"));
    $("#android-devices").innerHTML = android.devices.length ? android.devices.map(device =>
      `<div class="job"><strong>${escapeHtml(device.model || device.serial)}</strong><span class="state">${escapeHtml(device.state)}</span><p>${escapeHtml(device.serial)}${device.product ? ` · ${escapeHtml(device.product)}` : ""}</p></div>`
    ).join("") : `<p class="empty">${escapeHtml(android.issue || "No Android devices connected.")}</p>`;
    $("#compatibility-list").innerHTML = results.length ? results.map(item =>
      `<div class="job"><strong>${escapeHtml(item.result_type.replace("_", " "))} · ${escapeHtml(item.quantization)}</strong><span class="state">${item.passed ? "passed" : "failed"}</span><p>${escapeHtml(item.runtime)} · ${new Date(item.created_at).toLocaleString()}</p></div>`
    ).join("") : '<p class="empty">No results recorded yet.</p>';
  } catch (error) {
    $("#android-devices").innerHTML = `<p class="empty">${escapeHtml(error.message)}</p>`;
  }
}

function escapeHtml(value) { const node=document.createElement("span"); node.textContent=value; return node.innerHTML; }
async function uploadDataset(event) {
  event.preventDefault();
  const file = $("#dataset-file").files[0];
  const button = $("button", event.currentTarget);
  if (!file) return;
  button.disabled = true; button.dataset.label = button.textContent; button.textContent = "Uploading…";
  try {
    const query = new URLSearchParams({filename:file.name});
    for (const [key, value] of new FormData(event.currentTarget)) {
      if (key.endsWith("_field") && String(value).trim()) query.set(key, String(value).trim());
    }
    const response = await fetch(`/api/datasets/upload?${query}`, {
      method: "POST", headers: {"Content-Type": "application/octet-stream"}, body: file
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || `Upload failed (${response.status})`);
    $("#dataset-inspect-path").value = body.path;
    $("#training-dataset-path").value = body.path;
    showResult(body, "Dataset ready for training");
  } catch (error) {
    showResult({error:error.message}, "Upload needs attention");
  } finally {
    button.disabled = false; button.textContent = button.dataset.label;
  }
}
$$("[data-view]").forEach(b => b.onclick = () => showView(b.dataset.view));
$$("[data-go]").forEach(b => b.onclick = () => showView(b.dataset.go));
$$(".api-form").forEach(form => form.onsubmit = event => { event.preventDefault(); request(form.dataset.endpoint, formPayload(form), $("button", form)); });
$$("[data-config-action]").forEach(button => button.onclick = () => {
  const form = button.closest("form");
  const payload = form.id === "training-form" ? trainingPayload(form) : conversionPayload(form);
  request(button.dataset.configAction, payload, button);
});
$("#close-dialog").onclick = () => dialog.close();
$("#refresh").onclick = loadStatus;
$("#reload-jobs").onclick = loadJobs;
$("#reload-deployment").onclick = loadDeployment;
$("#reload-compatibility").onclick = loadDeployment;
$("#dataset-upload-form").onsubmit = uploadDataset;
loadStatus(); loadJobs(); setInterval(loadJobs, 5000);
