const fillMap = { 0: '0%', 1: '33.3%', 2: '66.6%', 3: '100%' };
const steps = document.querySelectorAll('.step');
const fill = document.getElementById('stepperFill');

steps.forEach(step => {
  step.addEventListener('click', () => {
    steps.forEach(s => s.classList.remove('active'));
    step.classList.add('active');
    fill.style.width = fillMap[step.dataset.step];
  });
});

// Drag & drop functionality
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');

dropzone.addEventListener('dragover', e => {
  e.preventDefault();
  dropzone.classList.add('dragover');
});

dropzone.addEventListener('dragleave', () => {
  dropzone.classList.remove('dragover');
});

dropzone.addEventListener('drop', e => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

// Toast notification
function showToast(msg, type = 'success') {
  const toast = document.getElementById('toast');
  const toastText = toast.querySelector('.toast-text');
  const toastIcon = toast.querySelector('.toast-icon');

  toastText.textContent = msg;

  if (type === 'error') {
    toastIcon.innerHTML = '<path stroke-linecap="round" stroke-linejoin="round" d="M6 18 18 6M6 6l12 12" />';
    toastIcon.style.color = '#ef4444';
  } else if (type === 'warning') {
    toastIcon.innerHTML = '<path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z" />';
    toastIcon.style.color = '#f59e0b';
  } else {
    toastIcon.innerHTML = '<path stroke-linecap="round" stroke-linejoin="round" d="m4.5 12.75 6 6 9-13.5" />';
    toastIcon.style.color = '#10b981';
  }

  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 3500);
}

// Download report functionality
function downloadReport(bestModel, bestScore, report) {
  const timestamp = new Date().toLocaleString();
  const content = `AutoML Report\n${'='.repeat(80)}\n\nGenerated: ${timestamp}\n\nBest Model: ${bestModel}\nBest Score: ${bestScore}\n\n${'='.repeat(80)}\n\n${report}`;
  const blob = new Blob([content], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `automl-report-${Date.now()}.txt`;
  a.click();
  URL.revokeObjectURL(url);
  showToast('Report downloaded successfully', 'success');
}

// Export results as JSON
function exportJSON(data) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `automl-results-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
  showToast('Results exported as JSON', 'success');
}

let currentFile = null;
let currentResults = null;

function getApiBaseUrl() {
  if (window.location.protocol.startsWith('http')) {
    const isLocalHost = ['localhost', '127.0.0.1'].includes(window.location.hostname);
    if (isLocalHost) {
      return 'http://127.0.0.1:8001';
    }
    return window.location.origin;
  }
  return 'http://127.0.0.1:8001';
}

// File handling
function handleFile(file) {
  currentFile = file;
  const allowed = ['csv', 'xlsx', 'xls', 'json'];
  const ext = file.name.split('.').pop().toLowerCase();

  if (!allowed.includes(ext)) {
    showToast('Unsupported file type. Use CSV, Excel, or JSON.', 'error');
    return;
  }

  const sizeMB = (file.size / 1024 / 1024).toFixed(1);
  showToast(`"${file.name}" loaded (${sizeMB} MB) — ready to configure.`, 'success');

  if (ext === 'csv') {
    const reader = new FileReader();
    reader.onload = function (e) {
      const text = e.target.result;
      const firstLine = text.split(/\n/)[0];
      const cols = firstLine.split(',');
      const select = document.getElementById('targetSelect');
      select.innerHTML = cols.map(c => `<option value="${c.trim()}">${c.trim()}</option>`).join('');
      document.getElementById('configSection').classList.remove('hidden');
    };
    reader.readAsText(file);
  } else {
    const select = document.getElementById('targetSelect');
    select.innerHTML = '<option value="">Select target column after upload</option>';
    document.getElementById('configSection').classList.remove('hidden');
  }

  steps.forEach(s => s.classList.remove('active'));
  steps[1].classList.add('active');
  fill.style.width = fillMap[1];
}

// Run AutoML
document.getElementById('runBtn').addEventListener('click', async () => {
  if (!currentFile) {
    showToast('Please select a file first.', 'warning');
    return;
  }

  const target = document.getElementById('targetSelect').value;
  if (!target) {
    showToast('Please select a target column.', 'warning');
    return;
  }

  steps.forEach(s => s.classList.remove('active'));
  steps[2].classList.add('active');
  fill.style.width = fillMap[2];

  const outputEl = document.getElementById('output');
  const outputContainer = document.querySelector('.output-container');
  outputEl.style.display = 'block';

  outputContainer.innerHTML = `
    <div class="loading-state">
      <div class="loading-spinner"></div>
      <div class="loading-text">Running AutoML... This may take a few minutes</div>
    </div>
  `;

  const form = new FormData();
  form.append('file', currentFile);
  form.append('target_column', target);

  try {
    const resp = await fetch(`${getApiBaseUrl()}/automl`, {
      method: 'POST',
      body: form,
      headers: { 'Accept': 'application/json' }
    });

    if (!resp.ok) {
      const errorData = await resp.json().catch(() => ({}));
      throw new Error(errorData.error || `HTTP Error: ${resp.status}`);
    }

    const data = await resp.json();
    currentResults = data;
    renderResults(data);

    steps.forEach(s => s.classList.remove('active'));
    steps[3].classList.add('active');
    fill.style.width = fillMap[3];

    showToast('AutoML training completed successfully!', 'success');
  } catch (err) {
    outputContainer.innerHTML = `
      <div class="loading-state">
        <div style="color:#ef4444;font-size:48px;margin-bottom:16px;">⚠️</div>
        <div style="color:#ef4444;font-size:18px;font-weight:600;margin-bottom:8px;">Error Running AutoML</div>
        <div style="color:#888;font-size:14px;">${err.message}</div>
      </div>
    `;
    showToast('Error running AutoML. Please try again.', 'error');
  }
});

function downloadBestModel(modelDownloadUrl) {
  if (!modelDownloadUrl) {
    showToast('Model download URL not available.', 'warning');
    return;
  }

  const normalizedUrl = modelDownloadUrl.startsWith('http')
    ? modelDownloadUrl
    : `${getApiBaseUrl()}${modelDownloadUrl.startsWith('/') ? '' : '/'}${modelDownloadUrl}`;

  const anchor = document.createElement('a');
  anchor.href = normalizedUrl;
  anchor.download = '';
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);

  showToast('Downloading best model...', 'success');
}

// ── SVG icon helpers ─────────────────────────────────────────────────────────
// Keeping SVGs in JS functions avoids any risk of broken inline strings

function iconDownload() {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M16.5 12 12 16.5m0 0L7.5 12m4.5 4.5V3"/></svg>`;
}

function iconUpload() {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5"/></svg>`;
}

function iconChart() {
  return `<svg class="section-icon" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 0 1 3 19.875v-6.75ZM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V8.625ZM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V4.125Z"/></svg>`;
}

function iconDoc() {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z"/></svg>`;
}

// ── Render results ────────────────────────────────────────────────────────────
function renderResults(data) {
  const outputContainer = document.querySelector('.output-container');
  const maxScore = Math.max(...Object.values(data.models));

  // Encode data safely for inline onclick handlers
  const safeReport = encodeURIComponent(data.report || '');
  const safeJSON = encodeURIComponent(JSON.stringify(data));

  // Build sorted model rows
  const sortedModels = Object.entries(data.models).sort((a, b) => b[1] - a[1]);

  const modelRows = sortedModels.map(([name, score]) => {
    const percentage = (score / maxScore) * 100;
    const isBest = name === data.best_model;
    return `
      <div class="model-item${isBest ? ' best' : ''}" data-testid="model-${name.replace(/\s+/g, '-').toLowerCase()}">
        <div class="model-header">
          <div class="model-name">
            ${name}
            ${isBest ? '<span class="metric-badge">Best</span>' : ''}
          </div>
          <div class="model-score">${score.toFixed(4)}</div>
        </div>
        <div class="progress-bar">
          <div class="progress-fill" style="width:${percentage}%"></div>
        </div>
      </div>`;
  }).join('');

  // Optional report section
  const reportSection = data.report ? `
    <div class="report-section">
      <div class="report-header">
        <h4>${iconDoc()} Detailed Report</h4>
      </div>
      <div class="report-content" data-testid="classification-report">${data.report}</div>
    </div>` : '';

  // Write final HTML — all tags are properly closed, no spaces inside < >
  outputContainer.innerHTML = `
    <div class="result-header">
      <div class="result-header-content">
        <h3>🎉 Training Complete</h3>
        <p>Your models have been trained and evaluated successfully</p>
      </div>
      <div class="header-actions">
        <button class="btn-download" data-testid="download-model-btn"
          onclick="downloadBestModel('${data.model_download_url || ''}')">
          ${iconDownload()} Download Model (.pkl)
        </button>
        <button class="btn-download" data-testid="download-report-btn"
          onclick="downloadReport('${data.best_model}', '${data.best_score.toFixed(4)}', decodeURIComponent('${safeReport}'))">
          ${iconDownload()} Download Report
        </button>
        <button class="btn-export" data-testid="export-json-btn"
          onclick="exportJSON(JSON.parse(decodeURIComponent('${safeJSON}')))">
          ${iconUpload()} Export JSON
        </button>
      </div>
    </div>

    <div class="result-content">
      <div class="result-grid">
        <div class="metric-card" data-testid="best-model-card">
          <div class="metric-header">
            <div class="metric-icon winner">🏆</div>
            <div class="metric-label">Best Model</div>
          </div>
          <div class="metric-value">
            ${data.best_model}
            <span class="metric-badge">Winner</span>
          </div>
        </div>
        <div class="metric-card" data-testid="best-score-card">
          <div class="metric-header">
            <div class="metric-icon accuracy">📊</div>
            <div class="metric-label">Best Score</div>
          </div>
          <div class="metric-value">${data.best_score.toFixed(4)}</div>
        </div>
      </div>

      <div class="models-section">
        <h3 class="section-title">${iconChart()} Model Performance Comparison</h3>
        <div class="model-list">
          ${modelRows}
        </div>
      </div>

      ${reportSection}
    </div>
  `;
}
