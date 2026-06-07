const API = "";

const figureTitles = {
  model_comparison: "Model Comparison — MAE & RMSE on Test Data",
  actual_vs_predicted: "Actual vs Predicted Power Output",
  correlation_matrix: "Feature Correlation Matrix",
  features_vs_pe: "Input Features vs Power Output",
  residual_analysis: "Residual Analysis",
};

function formatModelName(name) {
  return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

async function fetchJSON(url, options) {
  const res = await fetch(API + url, options);
  if (!res.ok) throw new Error(`Request failed: ${res.status}`);
  return res.json();
}

async function loadHealth() {
  const data = await fetchJSON("/api/health");
  const badge = document.getElementById("status-badge");
  badge.textContent = data.models_trained ? "Models Ready" : "Training...";
  badge.classList.toggle("ok", data.models_trained);
}

async function loadModels() {
  const data = await fetchJSON("/api/models");
  const select = document.getElementById("model-select");
  data.models.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = formatModelName(m) + (m === data.best_model ? " (best)" : "");
    select.appendChild(opt);
  });
}

async function loadMetrics() {
  const data = await fetchJSON("/api/metrics");
  const best = data.results[data.best_model];

  document.getElementById("metrics-summary").innerHTML = `
    <div class="stat-box"><div class="value">${data.train_samples.toLocaleString()}</div><div class="label">Training Samples</div></div>
    <div class="stat-box"><div class="value">${data.test_samples.toLocaleString()}</div><div class="label">Test Samples</div></div>
    <div class="stat-box"><div class="value">${best.mae}</div><div class="label">Best MAE (MW)</div></div>
    <div class="stat-box"><div class="value">${best.rmse}</div><div class="label">Best RMSE (MW)</div></div>
  `;

  const tbody = document.querySelector("#metrics-table tbody");
  tbody.innerHTML = "";
  Object.entries(data.results).forEach(([key, r]) => {
    const tr = document.createElement("tr");
    if (key === data.best_model) tr.classList.add("best");
    tr.innerHTML = `
      <td>${formatModelName(r.name)}</td>
      <td>${r.mae}</td>
      <td>${r.rmse}</td>
      <td>${r.r2}</td>
    `;
    tbody.appendChild(tr);
  });
}

async function loadFigures() {
  const data = await fetchJSON("/api/figures");
  const grid = document.getElementById("figures-grid");
  grid.innerHTML = "";

  data.figures.forEach((filename) => {
    const key = filename.replace(".png", "");
    const title = figureTitles[key] || key.replace(/_/g, " ");
    const figure = document.createElement("figure");
    figure.className = "figure-item";
    figure.innerHTML = `
      <img src="/api/figures/${filename}" alt="${title}" loading="lazy" />
      <figcaption>${title}</figcaption>
    `;
    grid.appendChild(figure);
  });
}

document.getElementById("predict-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = e.target;
  const body = {
    AT: parseFloat(form.AT.value),
    V: parseFloat(form.V.value),
    AP: parseFloat(form.AP.value),
    RH: parseFloat(form.RH.value),
  };
  const model = form.model.value;
  if (model) body.model = model;

  try {
    const result = await fetchJSON("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    document.getElementById("pe-value").textContent = result.predicted_pe;
    document.getElementById("model-used").textContent = formatModelName(result.model_used);
    document.getElementById("prediction-result").classList.remove("hidden");
  } catch (err) {
    alert("Prediction failed. Check your inputs and try again.");
    console.error(err);
  }
});

async function init() {
  try {
    await loadHealth();
    await Promise.all([loadModels(), loadMetrics(), loadFigures()]);
  } catch (err) {
    document.getElementById("status-badge").textContent = "Backend Offline";
    console.error(err);
  }
}

init();
