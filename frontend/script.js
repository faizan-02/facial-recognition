const API = window.location.origin;

document.addEventListener("DOMContentLoaded", () => {
    checkHealth();
    setupTabs();
    setupRegister();
    setupImageRecognition();
    setupVideoRecognition();
    loadFaces();

    document.getElementById("frame-skip").addEventListener("input", (e) => {
        const v = parseInt(e.target.value);
        document.getElementById("frame-skip-val").textContent =
            v === 0 ? "Process every frame" : `Process every ${v + 1}${ordinal(v + 1)} frame`;
    });
});

function ordinal(n) {
    const s = ["th", "st", "nd", "rd"];
    const v = n % 100;
    return s[(v - 20) % 10] || s[v] || s[0];
}

async function checkHealth() {
    const badge = document.getElementById("status");
    try {
        const res = await fetch(`${API}/health`);
        const data = await res.json();
        badge.textContent = `Online | ${data.registered_faces} faces`;
        badge.className = "status-badge online";
    } catch {
        badge.textContent = "Offline";
        badge.className = "status-badge offline";
    }
}

function setupTabs() {
    document.querySelectorAll(".tab").forEach((tab) => {
        tab.addEventListener("click", () => {
            document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
            document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
            tab.classList.add("active");
            document.getElementById(`tab-${tab.dataset.tab}`).classList.add("active");
            if (tab.dataset.tab === "database") loadFaces();
        });
    });
}

function setupUploadArea(areaId, fileId, previewId, onFile) {
    const area = document.getElementById(areaId);
    const input = document.getElementById(fileId);

    area.addEventListener("click", () => input.click());
    area.addEventListener("dragover", (e) => { e.preventDefault(); area.classList.add("dragover"); });
    area.addEventListener("dragleave", () => area.classList.remove("dragover"));
    area.addEventListener("drop", (e) => {
        e.preventDefault();
        area.classList.remove("dragover");
        if (e.dataTransfer.files.length) {
            input.files = e.dataTransfer.files;
            handleFileSelect(input, previewId, onFile);
        }
    });
    input.addEventListener("change", () => handleFileSelect(input, previewId, onFile));
}

function handleFileSelect(input, previewId, onFile) {
    const file = input.files[0];
    if (!file) return;
    if (previewId) {
        const preview = document.getElementById(previewId);
        if (preview && file.type.startsWith("image/")) {
            preview.src = URL.createObjectURL(file);
            preview.hidden = false;
            preview.previousElementSibling.hidden = true;
        }
    }
    if (onFile) onFile(file);
}

function setupRegister() {
    setupUploadArea("reg-upload", "reg-file", "reg-preview", null);

    document.getElementById("register-form").addEventListener("submit", async (e) => {
        e.preventDefault();
        const name = document.getElementById("reg-name").value.trim();
        const file = document.getElementById("reg-file").files[0];
        if (!name || !file) return;

        showLoading("Registering face...");
        const form = new FormData();
        form.append("name", name);
        form.append("file", file);

        try {
            const res = await fetch(`${API}/api/register`, { method: "POST", body: form });
            const data = await res.json();
            const box = document.getElementById("reg-result");
            box.hidden = false;

            if (res.ok) {
                box.className = "result-box success";
                box.innerHTML = `<strong>Success!</strong> ${data.message}`;
                document.getElementById("reg-name").value = "";
                document.getElementById("reg-file").value = "";
                document.getElementById("reg-preview").hidden = true;
                document.querySelector("#reg-upload .upload-placeholder").hidden = false;
                checkHealth();
            } else {
                box.className = "result-box error";
                box.innerHTML = `<strong>Error:</strong> ${data.detail || data.message}`;
            }
        } catch (err) {
            showError("reg-result", err.message);
        }
        hideLoading();
    });
}

function setupImageRecognition() {
    let selectedFile = null;
    setupUploadArea("img-upload", "img-file", "img-preview", (file) => {
        selectedFile = file;
        document.getElementById("img-submit").disabled = false;
    });

    document.getElementById("img-submit").addEventListener("click", async () => {
        if (!selectedFile) return;
        showLoading("Detecting and recognizing faces...");

        const form = new FormData();
        form.append("file", selectedFile);

        try {
            const res = await fetch(`${API}/api/recognize/image`, { method: "POST", body: form });
            const data = await res.json();
            const box = document.getElementById("img-result");
            box.hidden = false;

            if (res.ok) {
                box.className = "result-box info";
                let html = `<strong>${data.total_faces} face(s) detected</strong><br>`;
                data.faces.forEach((f, i) => {
                    const id = f.identity ? `<span style="color:var(--green)">${f.identity}</span> (${(f.similarity * 100).toFixed(0)}%)` : `<span style="color:var(--orange)">Unknown</span>`;
                    html += `Face ${i + 1}: ${id} | Conf: ${(f.confidence * 100).toFixed(0)}%`;
                    if (f.age) html += ` | Age: ~${f.age}`;
                    if (f.gender) html += ` | ${f.gender}`;
                    html += `<br>`;
                });
                box.innerHTML = html;

                const container = document.getElementById("img-annotated-container");
                container.hidden = false;
                document.getElementById("img-annotated").src = `${API}${data.annotated_image}`;
            } else {
                box.className = "result-box error";
                box.innerHTML = `<strong>Error:</strong> ${data.detail}`;
            }
        } catch (err) {
            showError("img-result", err.message);
        }
        hideLoading();
    });
}

function setupVideoRecognition() {
    let selectedFile = null;
    const area = document.getElementById("vid-upload");
    const input = document.getElementById("vid-file");

    area.addEventListener("click", () => input.click());
    area.addEventListener("dragover", (e) => { e.preventDefault(); area.classList.add("dragover"); });
    area.addEventListener("dragleave", () => area.classList.remove("dragover"));
    area.addEventListener("drop", (e) => {
        e.preventDefault();
        area.classList.remove("dragover");
        if (e.dataTransfer.files.length) { input.files = e.dataTransfer.files; onVidFile(); }
    });
    input.addEventListener("change", onVidFile);

    function onVidFile() {
        selectedFile = input.files[0];
        if (selectedFile) {
            document.getElementById("vid-submit").disabled = false;
            area.querySelector(".upload-placeholder").innerHTML =
                `<span class="upload-icon">&#9654;</span><span>${selectedFile.name} (${(selectedFile.size / 1048576).toFixed(1)} MB)</span>`;
        }
    }

    document.getElementById("vid-submit").addEventListener("click", async () => {
        if (!selectedFile) return;
        const progress = document.getElementById("vid-progress");
        progress.hidden = false;
        showLoading("Processing video — this may take a while on CPU...");

        const form = new FormData();
        form.append("file", selectedFile);
        form.append("frame_skip", document.getElementById("frame-skip").value);

        try {
            const res = await fetch(`${API}/api/recognize/video`, { method: "POST", body: form });
            const data = await res.json();
            progress.hidden = true;
            const box = document.getElementById("vid-result");
            box.hidden = false;

            if (res.ok) {
                box.className = "result-box info";
                box.innerHTML = `
                    <strong>Video processed!</strong><br>
                    Frames: ${data.total_frames} total, ${data.processed_frames} analyzed<br>
                    Avg faces/frame: ${data.avg_faces_per_frame}
                `;
                const container = document.getElementById("vid-output-container");
                container.hidden = false;
                document.getElementById("vid-output").src = `${API}${data.annotated_video}`;
            } else {
                box.className = "result-box error";
                box.innerHTML = `<strong>Error:</strong> ${data.detail}`;
            }
        } catch (err) {
            progress.hidden = true;
            showError("vid-result", err.message);
        }
        hideLoading();
    });
}

async function loadFaces() {
    try {
        const res = await fetch(`${API}/api/faces`);
        const faces = await res.json();
        const grid = document.getElementById("faces-list");
        const empty = document.getElementById("no-faces");

        if (!faces.length) {
            grid.innerHTML = "";
            empty.hidden = false;
            return;
        }

        empty.hidden = true;
        grid.innerHTML = faces.map((f) => `
            <div class="face-card">
                <div class="avatar">${f.name.charAt(0).toUpperCase()}</div>
                <div class="name">${f.name}</div>
                <div class="samples">${f.samples} sample(s)</div>
                <button class="btn btn-danger" onclick="deleteFace('${f.name}')">Remove</button>
            </div>
        `).join("");
    } catch {}
}

async function deleteFace(name) {
    if (!confirm(`Delete "${name}" from the database?`)) return;
    try {
        await fetch(`${API}/api/faces/${encodeURIComponent(name)}`, { method: "DELETE" });
        loadFaces();
        checkHealth();
    } catch {}
}

function showLoading(text) {
    document.getElementById("loading-text").textContent = text;
    document.getElementById("loading-overlay").hidden = false;
}

function hideLoading() {
    document.getElementById("loading-overlay").hidden = true;
}

function showError(boxId, message) {
    const box = document.getElementById(boxId);
    box.hidden = false;
    box.className = "result-box error";
    box.innerHTML = `<strong>Error:</strong> ${message}`;
}
