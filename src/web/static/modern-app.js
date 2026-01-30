const folderPathInput = document.getElementById("folderPath");
const folderPicker = document.getElementById("folderPicker");
const pickedFolder = document.getElementById("pickedFolder");
const runButton = document.getElementById("runPipeline");
const statusText = document.getElementById("status");
const treeView = document.getElementById("treeView");
const warningsList = document.getElementById("warnings");
const attentionList = document.getElementById("attentionList");
const suggestionsContainer = document.getElementById("suggestions");
const issuesContainer = document.getElementById("issues");
const summaryTotal = document.getElementById("summaryTotal");
const summaryLow = document.getElementById("summaryLow");
const summaryWarnings = document.getElementById("summaryWarnings");
const summaryDuplicates = document.getElementById("summaryDuplicates");

folderPicker.addEventListener("change", () => {
    if (!folderPicker.files.length) {
        pickedFolder.textContent = "";
        return;
    }
    const relativePath = folderPicker.files[0].webkitRelativePath || "";
    const folderName = relativePath.split("/")[0];
    pickedFolder.textContent = folderName ? `Dossier sélectionné : ${folderName}` : "Dossier sélectionné.";
    if (!folderPathInput.value && folderName) {
        folderPathInput.value = folderName;
    }
});

let statusInterval = null;

function pollStatus() {
    clearInterval(statusInterval);
    statusInterval = setInterval(async () => {
        try {
            const res = await fetch("/status");
            const text = await res.text();
            statusText.textContent = text;
            if (text === "Terminé") clearInterval(statusInterval);
        } catch {
            // ignore
        }
    }, 1000);
}

runButton.addEventListener("click", async () => {
    const folderPath = folderPathInput.value.trim();
    if (!folderPath) {
        setLoading(false, "⚠️ Veuillez fournir un chemin de dossier valide.", true);
        return;
    }

    setLoading(true, "🔄 Analyse en cours... Cela peut prendre plusieurs minutes.");
    pollStatus();

    try {
        const response = await fetch("/run_pipeline", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ folder_path: folderPath }),
        });

        if (!response.ok) {
            const errorBody = await response.json();
            throw new Error(errorBody.detail || "Erreur inconnue.");
        }

        const payload = await response.json();
        renderResults(payload.data);
        setLoading(false, "✅ Analyse terminée avec succès!");
    } catch (error) {
        setLoading(false, `❌ Erreur: ${error.message}`, true);
    }
});

function setLoading(isLoading, message, isError = false) {
    runButton.disabled = isLoading;
    statusText.textContent = message;
    statusText.style.display = 'flex';
    statusText.className = isLoading ? 'status loading' : 'status';
    statusText.style.borderLeftColor = isError ? 'var(--danger)' : 'var(--primary)';
}

function renderResults(data) {
    const { hierarchy, categorizations, reviewer } = data;
    const confidenceMap = new Map(categorizations.map((item) => [item.filename, item]));

    updateSummary(categorizations, hierarchy.warnings || [], reviewer?.issues || []);
    renderAttention(reviewer?.issues || []);
    renderTree(hierarchy.folder_structure, confidenceMap);
    renderWarnings(hierarchy.warnings || [], reviewer?.issues || []);
    renderSuggestions(reviewer?.suggestions || []);
    renderIssues(reviewer?.issues || []);
}

function updateSummary(categorizations, warnings, issues) {
    const lowConfidence = categorizations.filter((item) => item.confidence < 0.55);
    const duplicateIssue = issues.find((issue) => issue.issue_type === "possible_duplicates");

    animateValue(summaryTotal, categorizations.length);
    animateValue(summaryLow, lowConfidence.length);
    animateValue(summaryWarnings, warnings.length);
    animateValue(summaryDuplicates, duplicateIssue ? duplicateIssue.affected_files.length : 0);
}

function animateValue(element, targetValue) {
    const duration = 1000;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const current = Math.floor(progress * targetValue);
        element.textContent = current.toString();

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

function renderAttention(issues) {
    attentionList.innerHTML = "";
    if (!issues.length) {
        attentionList.classList.add("empty");
        attentionList.innerHTML = "<li>Aucun point critique détecté</li>";
        return;
    }

    attentionList.classList.remove("empty");
    issues.forEach((issue, index) => {
        const li = document.createElement("li");
        li.className = "attention";
        li.style.animationDelay = `${index * 0.1}s`;
        const affected = issue.affected_files.length
            ? issue.affected_files.slice(0, 6).join(", ")
            : "Aucun fichier spécifique";
        li.innerHTML = `
            <strong>${issue.issue_type} (${issue.severity})</strong>
            <p>${issue.description}</p>
            <p><b>Exemples :</b> ${affected}</p>
        `;
        attentionList.appendChild(li);
    });
}

function renderTree(folderStructure, confidenceMap) {
    treeView.innerHTML = "";
    const treeData = buildTree(folderStructure);
    const entries = Object.entries(treeData);
    if (!entries.length) {
        treeView.innerHTML = '<p style="text-align: center; color: var(--text-muted); padding: 2rem;">Aucune hiérarchie disponible</p>';
        return;
    }

    const rootList = document.createElement("ul");
    entries.forEach(([folder, node]) => {
        rootList.appendChild(renderNode(folder, node, confidenceMap));
    });
    treeView.appendChild(rootList);
}

function buildTree(folderStructure) {
    const tree = {};
    Object.entries(folderStructure).forEach(([folderPath, files]) => {
        const parts = folderPath.split("/");
        let current = tree;
        parts.forEach((part) => {
            if (!current[part]) {
                current[part] = { __files: [], __children: {} };
            }
            current = current[part].__children;
        });
        const node = getNode(tree, parts);
        node.__files.push(...files);
    });
    return tree;
}

function getNode(tree, parts) {
    let current = tree;
    let node = null;
    parts.forEach((part) => {
        node = current[part];
        current = node.__children;
    });
    return node;
}

function renderNode(name, node, confidenceMap) {
    const li = document.createElement("li");
    const label = document.createElement("span");
    label.className = "node";
    label.textContent = `📁 ${name}`;
    li.appendChild(label);

    const childrenList = document.createElement("ul");

    Object.entries(node.__children).forEach(([childName, childNode]) => {
        childrenList.appendChild(renderNode(childName, childNode, confidenceMap));
    });

    node.__files.forEach((fileName) => {
        const fileLi = document.createElement("li");
        const fileTag = document.createElement("span");
        fileTag.className = "file";

        const confidenceData = confidenceMap.get(fileName);
        const confidence = confidenceData?.confidence ?? 0;
        const tag = document.createElement("span");
        tag.className = `tag ${getConfidenceLevel(confidence)}`;
        tag.textContent = `${(confidence * 100).toFixed(0)}%`;

        fileTag.innerHTML = `📄 ${fileName}`;
        fileTag.appendChild(tag);

        fileLi.appendChild(fileTag);
        childrenList.appendChild(fileLi);
    });

    if (childrenList.children.length) {
        li.appendChild(childrenList);
    }

    return li;
}

function getConfidenceLevel(confidence) {
    if (confidence >= 0.8) return "high";
    if (confidence >= 0.5) return "medium";
    return "low";
}

function renderWarnings(hierarchyWarnings, issues) {
    warningsList.innerHTML = "";
    const warnings = [...hierarchyWarnings];
    const issueSummaries = issues.map((issue) => `${issue.issue_type} (${issue.severity})`);
    const combined = [...warnings, ...issueSummaries];

    if (!combined.length) {
        warningsList.classList.add("empty");
        warningsList.innerHTML = "<li>Aucun avertissement</li>";
        return;
    }

    warningsList.classList.remove("empty");
    combined.forEach((warning, index) => {
        const li = document.createElement("li");
        li.textContent = warning;
        li.style.animationDelay = `${index * 0.05}s`;
        warningsList.appendChild(li);
    });
}

function renderSuggestions(suggestions) {
    suggestionsContainer.innerHTML = "";
    if (!suggestions.length) {
        suggestionsContainer.classList.add("empty");
        suggestionsContainer.textContent = "Aucune suggestion disponible";
        return;
    }

    suggestionsContainer.classList.remove("empty");
    suggestions.forEach((suggestion, index) => {
        const box = document.createElement("div");
        box.className = "suggestion";
        box.style.animationDelay = `${index * 0.1}s`;
        box.innerHTML = `
            <strong>${suggestion.action}</strong>
            <p><b>Cible :</b> ${suggestion.target}</p>
            <p>${suggestion.suggestion}</p>
        `;
        suggestionsContainer.appendChild(box);
    });
}

function renderIssues(issues) {
    issuesContainer.innerHTML = "";
    if (!issues.length) {
        issuesContainer.classList.add("empty");
        issuesContainer.textContent = "Aucun problème détecté";
        return;
    }

    issuesContainer.classList.remove("empty");
    issues.forEach((issue, index) => {
        const box = document.createElement("div");
        box.className = "issue";
        box.style.animationDelay = `${index * 0.1}s`;
        const files = issue.affected_files.length
            ? issue.affected_files.join(", ")
            : "Aucun fichier spécifique";
        box.innerHTML = `
            <strong>${issue.severity}</strong>
            <p><b>Type :</b> ${issue.issue_type}</p>
            <p>${issue.description}</p>
            <p><b>Fichiers :</b> ${files}</p>
        `;
        issuesContainer.appendChild(box);
    });
}