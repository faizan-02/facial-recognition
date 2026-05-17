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
            v === 0 ? "Process every frame (slowest, most accurate)" : `Process every ${v + 1}${ordinal(v + 1)} frame`;
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
        badge.textContent = `Online | ${data.registered_identities} identities, ${data.total_samples} samples`;
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
            const placeholder = preview.previousElementSibling;
            if (placeholder) placeholder.hidden = true;
        }
    }
    if (onFile) onFile(file);
}

function setupRegister() {
    const area = document.getElementById("reg-upload");
    const input = document.getElementById("reg-file");
    const previewGrid = document.getElementById("reg-previews");

    area.addEventListener("click", () => input.click());
    area.addEventListener("dragover", (e) => { e.preventDefault(); area.classList.add("dragover"); });
    area.addEventListener("dragleave", () => area.classList.remove("dragover"));
    area.addEventListener("drop", (e) => {
        e.preventDefault();
        area.classList.remove("dragover");
        if (e.dataTransfer.files.length) {
            input.files = e.dataTransfer.files;
            showRegPreviews();
        }
    });
    input.addEventListener("change", showRegPreviews);

    function showRegPreviews() {
        previewGrid.innerHTML = "";
        for (const file of input.files) {
            if (!file.type.startsWith("image/")) continue;
            const div = document.createElement("div");
            div.className = "preview-thumb";
            const img = document.createElement("img");
            img.src = URL.createObjectURL(file);
            const label = document.createElement("span");
            label.textContent = file.name.substring(0, 20);
            div.appendChild(img);
            div.appendChild(label);
            previewGrid.appendChild(div);
        }
    }

    document.getElementById("register-form").addEventListener("submit", async (e) => {
        e.preventDefault();
        const name = document.getElementById("reg-name").value.trim();
        const files = input.files;
        if (!name || !files.length) return;

        showLoading(`Registering ${files.length} image(s)...`);
        const box = document.getElementById("reg-result");

        try {
            if (files.length === 1) {
                const form = new FormData();
                form.append("name", name);
                form.append("file", files[0]);
                const res = await fetch(`${API}/api/register`, { method: "POST", body: form });
                const data = await res.json();
                box.hidden = false;
                if (res.ok) {
                    box.className = "result-box success";
                    box.innerHTML = `<strong>Success!</strong> ${data.message}<br>Poses registered: ${(data.poses_registered || []).join(", ")}`;
                } else {
                    box.className = "result-box error";
                    box.innerHTML = `<strong>Error:</strong> ${data.detail || data.message}`;
                }
            } else {
                const form = new FormData();
                form.append("name", name);
                for (const file of files) form.append("files", file);
                const res = await fetch(`${API}/api/register/multi`, { method: "POST", body: form });
                const data = await res.json();
                box.hidden = false;
                if (res.ok) {
                    box.className = "result-box success";
                    let html = `<strong>${data.registered}/${data.total} images registered.</strong><br>`;
                    data.results.forEach((r, i) => {
                        const icon = r.success ? "&#10003;" : "&#10007;";
                        html += `${icon} Image ${i + 1}: ${r.message}<br>`;
                    });
                    box.innerHTML = html;
                } else {
                    box.className = "result-box error";
                    box.innerHTML = `<strong>Error:</strong> ${data.detail}`;
                }
            }
            document.getElementById("reg-name").value = "";
            input.value = "";
            previewGrid.innerHTML = "";
            checkHealth();
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
        showLoading("Running multi-scale detection & recognition...");

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
                    const id = f.identity
                        ? `<span style="color:var(--green)">${f.identity}</span> (${(f.similarity * 100).toFixed(0)}%)`
                        : `<span style="color:var(--orange)">Unknown</span>`;
                    html += `Face ${i + 1}: ${id} | Conf: ${(f.confidence * 100).toFixed(0)}%`;
                    if (f.age) html += ` | Age: ~${f.age}`;
                    if (f.gender) html += ` | ${f.gender}`;
                    if (f.pose && f.pose !== "frontal") html += ` | ${f.pose}`;
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
        document.getElementById("vid-submit").disabled = true;
        document.getElementById("vid-result").hidden = true;
        document.getElementById("vid-output-container").hidden = true;

        const form = new FormData();
        form.append("file", selectedFile);
        form.append("frame_skip", document.getElementById("frame-skip").value);

        try {
            const res = await fetch(`${API}/api/recognize/video`, { method: "POST", body: form });
            const data = await res.json();

            if (data.job_id) {
                pollVideoJob(data.job_id);
            } else {
                progress.hidden = true;
                showError("vid-result", data.detail || "Failed to start processing");
            }
        } catch (err) {
            progress.hidden = true;
            showError("vid-result", err.message);
            document.getElementById("vid-submit").disabled = false;
        }
    });
}

async function pollVideoJob(jobId) {
    const progressFill = document.getElementById("vid-progress-fill");
    const progressText = document.getElementById("vid-progress-text");

    const poll = async () => {
        try {
            const res = await fetch(`${API}/api/recognize/video/${jobId}`);
            const data = await res.json();

            if (data.status === "processing") {
                progressFill.style.width = `${data.progress}%`;
                progressFill.style.animation = "none";
                progressText.textContent = `Processing... ${data.progress.toFixed(1)}%`;
                setTimeout(poll, 1500);
            } else if (data.status === "done") {
                document.getElementById("vid-progress").hidden = true;
                document.getElementById("vid-submit").disabled = false;
                const box = document.getElementById("vid-result");
                box.hidden = false;
                box.className = "result-box info";
                const r = data.result;
                box.innerHTML = `
                    <strong>Video processed!</strong><br>
                    Frames: ${r.total_frames} total, ${r.processed_frames} analyzed<br>
                    Avg faces/frame: ${r.avg_faces_per_frame}<br>
                    Identified: ${r.unique_identities.length ? r.unique_identities.join(", ") : "none (register faces first)"}
                `;
                const container = document.getElementById("vid-output-container");
                container.hidden = false;
                document.getElementById("vid-output").src = `${API}${data.annotated_video}`;
            } else {
                document.getElementById("vid-progress").hidden = true;
                document.getElementById("vid-submit").disabled = false;
                showError("vid-result", data.error || "Processing failed");
            }
        } catch (err) {
            setTimeout(poll, 3000);
        }
    };
    poll();
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
        grid.innerHTML = faces.map((f) => {
            const coverageColor = {
                excellent: "var(--green)",
                good: "#6cb4ee",
                fair: "var(--orange)",
                poor: "var(--red)",
            }[f.coverage] || "var(--text-dim)";

            return `
                <div class="face-card">
                    <div class="avatar">${f.name.charAt(0).toUpperCase()}</div>
                    <div class="name">${f.name}</div>
                    <div class="samples">${f.samples} sample(s)</div>
                    <div class="poses">Poses: ${f.poses.join(", ") || "none"}</div>
                    <div class="coverage" style="color:${coverageColor}">Coverage: ${f.coverage}</div>
                    <button class="btn btn-danger" onclick="deleteFace('${f.name}')">Remove</button>
                </div>
            `;
        }).join("");
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
