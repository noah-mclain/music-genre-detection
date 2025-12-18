// State management
const state = {
  selectedFiles: [],
  isProcessing: false,
};

// DOM Elements
const uploadBox = document.getElementById("uploadBox");
const fileInput = document.getElementById("fileInput");
const selectBtn = document.getElementById("selectBtn");
const clearBtn = document.getElementById("clearBtn");
const analyzeBtn = document.getElementById("analyzeBtn");
const fileList = document.getElementById("fileList");
const selectedFilesList = document.getElementById("selectedFiles");
const progressSection = document.getElementById("progressSection");
const progressFill = document.getElementById("progressFill");
const progressText = document.getElementById("progressText");
const resultsSection = document.getElementById("resultsSection");
const resultsContainer = document.getElementById("resultsContainer");
const errorSection = document.getElementById("errorSection");
const errorText = document.getElementById("errorText");

// Initialize event listeners
function initializeEventListeners() {
  uploadBox.addEventListener("click", () => fileInput.click());
  uploadBox.addEventListener("dragover", handleDragOver);
  uploadBox.addEventListener("dragleave", handleDragLeave);
  uploadBox.addEventListener("drop", handleDrop);

  fileInput.addEventListener("change", handleFileSelect);
  selectBtn.addEventListener("click", () => fileInput.click());
  clearBtn.addEventListener("click", clearFiles);
  analyzeBtn.addEventListener("click", analyzeFiles);
}

// Handle drag over
function handleDragOver(e) {
  e.preventDefault();
  e.stopPropagation();
  uploadBox.classList.add("dragover");
}

// Handle drag leave
function handleDragLeave(e) {
  e.preventDefault();
  e.stopPropagation();
  uploadBox.classList.remove("dragover");
}

// Handle drop
function handleDrop(e) {
  e.preventDefault();
  e.stopPropagation();
  uploadBox.classList.remove("dragover");

  const files = Array.from(e.dataTransfer.files);
  addFiles(files);
}

// Handle file select
function handleFileSelect(e) {
  const files = Array.from(e.target.files);
  addFiles(files);
}

// Add files to state
function addFiles(files) {
  const validFiles = files.filter((file) => {
    const extension = file.name.split(".").pop().toLowerCase();
    return ["mp3", "wav", "m4a", "flac"].includes(extension);
  });

  if (validFiles.length === 0) {
    showError(
      "No valid audio files selected. Allowed formats: MP3, WAV, M4A, FLAC"
    );
    return;
  }
  validFiles.forEach((file) => {
    const isDuplicate = state.selectedFiles.some(
      (f) => f.name === file.name && f.size === file.size
    );
    if (!isDuplicate) {
      state.selectedFiles.push(file);
    }
  });
  updateFileList();
  updateButtons();
}

// Update file list display
function updateFileList() {
  if (state.selectedFiles.length === 0) {
    fileList.style.display = "none";
    return;
  }

  fileList.style.display = "block";
  selectedFilesList.innerHTML = "";

  state.selectedFiles.forEach((file, index) => {
    const li = document.createElement("li");
    li.innerHTML = `
            <div class="file-name">${escapeHtml(file.name)}</div>
            <div class="file-size">${formatFileSize(file.size)}</div>
            <button class="remove-file" onclick="removeFile(${index})" title="Remove">×</button>
        `;
    selectedFilesList.appendChild(li);
  });
}

// Remove file from list
function removeFile(index) {
  state.selectedFiles.splice(index, 1);
  updateFileList();
  updateButtons();
}

// Update button states
function updateButtons() {
  if (state.selectedFiles.length === 0) {
    fileList.style.display = "none";
    clearBtn.style.display = "none";
    analyzeBtn.style.display = "none";
    selectBtn.style.display = "block";
  } else {
    selectBtn.style.display = "block";
    clearBtn.style.display = "block";
    analyzeBtn.style.display = "block";
    analyzeBtn.disabled = false;
  }
}

// Clear all files
function clearFiles() {
  state.selectedFiles = [];
  fileInput.value = "";
  updateFileList();
  updateButtons();
  hideAll();
}

// Analyze files
async function analyzeFiles() {
  if (state.selectedFiles.length === 0) {
    showError("Please select files to analyze");
    return;
  }

  state.isProcessing = true;
  analyzeBtn.disabled = true;

  // Show progress section
  hideAll();
  progressSection.style.display = "block";

  try {
    // Create FormData
    const formData = new FormData();
    state.selectedFiles.forEach((file) => {
      formData.append("files", file);
    });

    // Send request
    const response = await fetch("/upload", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();

    // Update progress
    updateProgress(100, `${data.results.length} files processed`);

    // Display results
    await displayResults(data.results);
  } catch (error) {
    console.error("Error:", error);
    showError(`Error analyzing files: ${error.message}`);
  } finally {
    state.isProcessing = false;
    analyzeBtn.disabled = false;
    setTimeout(() => {
      progressSection.style.display = "none";
    }, 500);
  }
}

// Update progress bar
function updateProgress(percent, text) {
  progressFill.style.width = `${percent}%`;
  if (text) {
    progressText.textContent = text;
  }
}

// Display results
async function displayResults(results) {
  resultsContainer.innerHTML = "";

  results.forEach((result) => {
    const card = document.createElement("div");
    card.className = `result-card ${result.success ? "success" : "error"}`;

    if (result.success) {
      const confidence = result.confidence;
      const predictions = result.predictions || {};

      card.innerHTML = `
                <div class="result-filename">📁 ${escapeHtml(
                  result.filename
                )}</div>
                <div class="result-genre">🎵 ${escapeHtml(result.genre)}</div>
                <div class="result-confidence">
                    Confidence: ${confidence.toFixed(2)}%
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${confidence}%"></div>
                </div>
                ${renderPredictions(predictions)}
            `;
    } else {
      card.innerHTML = `
                <div class="result-filename">📁 ${escapeHtml(
                  result.filename
                )}</div>
                <div class="error-text">❌ Error: ${escapeHtml(
                  result.error
                )}</div>
            `;
    }

    resultsContainer.appendChild(card);
  });

  resultsSection.style.display = "block";
}

// Render all predictions
function renderPredictions(predictions) {
  if (!predictions || Object.keys(predictions).length === 0) {
    return "";
  }

  const entries = Object.entries(predictions)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5); // Top 5
  const html = entries
    .map(
      ([genre, score]) => `
        <div class="prediction-item">
            <span>${escapeHtml(genre)}</span>
            <div class="prediction-bar">
                <div class="prediction-bar-fill" style="width: ${
                  score * 100
                }%"></div>
            </div>
            <span>${(score * 100).toFixed(1)}%</span>
        </div>
    `
    )
    .join("");

  return `<div class="all-predictions"><h4>Top Predictions:</h4>${html}</div>`;
}

// Show error
function showError(message) {
  hideAll();
  errorText.textContent = message;
  errorSection.style.display = "block";
}

// Hide all sections
function hideAll() {
  progressSection.style.display = "none";
  resultsSection.style.display = "none";
  errorSection.style.display = "none";
}

// Utility: Format file size
function formatFileSize(bytes) {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + " " + sizes[i];
}

// Utility: Escape HTML
function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// Initialize on DOM ready
document.addEventListener("DOMContentLoaded", () => {
  initializeEventListeners();
  console.log("Music Genre Classifier loaded");
});
