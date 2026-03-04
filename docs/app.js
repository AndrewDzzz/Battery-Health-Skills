const skills = [
  {
    name: "feature-engineering",
    description:
      "Per-cycle mechanism-aware battery feature extraction for interpretable diagnostic models.",
    command: "python feature-engineering/scripts/extract_features.py",
    path: "feature-engineering/",
    tags: ["features", "soh", "fault-detection", "interpretability"],
  },
  {
    name: "interpretability-pipeline",
    description:
      "Train GBDT baselines, compute SHAP attributions, and create LLM-ready diagnostic prompts.",
    command: "python interpretability-pipeline/scripts/train_gbdt_shap.py",
    path: "interpretability-pipeline/",
    tags: ["explainer", "prompting", "modeling"],
  },
  {
    name: "battery-telemetry-integrity",
    description:
      "Detect replay-like patterns, duplicates, timestamp regressions, and impossible jumps before modeling.",
    command: "python battery-telemetry-integrity/scripts/check_telemetry_integrity.py",
    path: "battery-telemetry-integrity/",
    tags: ["security", "data-quality", "validation"],
  },
  {
    name: "battery-security-audit",
    description:
      "Generate risk registers and owner-driven control plans for telemetry, firmware, transport, and model-serving layers.",
    command: "python battery-security-audit/scripts/generate_security_audit.py",
    path: "battery-security-audit/",
    tags: ["threat-model", "governance", "risk"],
  },
  {
    name: "soh-modeling-upgrade",
    description:
      "Extract richer SOH indicators and train uncertainty-aware regressors with drift checks.",
    command: "python soh-modeling-upgrade/scripts/train_soh_with_uncertainty.py",
    path: "soh-modeling-upgrade/",
    tags: ["soh", "uncertainty", "drift"],
  },
  {
    name: "soh-field-demo",
    description:
      "Run a realistic EV fleet workflow: synthetic or real telemetry to integrity checks, SOH results, and security preflight.",
    command: "python soh-field-demo/scripts/run_battery_field_demo.py",
    path: "soh-field-demo/",
    tags: ["demo", "deployment", "end-to-end"],
  },
];

const skillGrid = document.getElementById("skill-grid");
const filterInput = document.getElementById("skill-filter");
const skillCount = document.getElementById("skill-count");
const clearFilter = document.getElementById("clear-filter");
const emptyState = document.getElementById("skills-empty");

function renderSkills(filter = "") {
  const q = filter.trim().toLowerCase();
  const nodes = skills.filter((skill) => {
    const haystack = `${skill.name} ${skill.description} ${skill.tags.join(" ")}`.toLowerCase();
    return !q || haystack.includes(q);
  });

  skillGrid.innerHTML = "";
  nodes.forEach((skill, idx) => {
    const card = document.createElement("article");
    card.className = "skill-card";
    card.style.animationDelay = `${idx * 0.05}s`;

    card.innerHTML = `
      <h3>${skill.name}</h3>
      <p>${skill.description}</p>
      <p><strong>Command:</strong> <code>${skill.command}</code></p>
      <div class="tags">${skill.tags.map((tag) => `<button class="tag" data-tag="${tag}">${tag}</button>`).join("")}</div>
      <a href="${skill.path}">Open skill folder</a>
    `;
    skillGrid.appendChild(card);
  });

  skillCount.textContent = `${nodes.length} of ${skills.length} skills`;
  if (nodes.length === 0) {
    emptyState.classList.add("active");
  } else {
    emptyState.classList.remove("active");
  }

  skillGrid.querySelectorAll(".tag").forEach((tagButton) => {
    tagButton.addEventListener("click", () => {
      const tag = tagButton.dataset.tag;
      filterInput.value = tag;
      renderSkills(tag);
      filterInput.focus();
    });

    tagButton.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        tagButton.click();
      }
    });
  });

  if (nodes.length === 0 && q) {
    filterInput.setAttribute("aria-invalid", "true");
  } else {
    filterInput.removeAttribute("aria-invalid");
  }
}

filterInput?.addEventListener("input", (event) => {
  renderSkills(event.target.value);
});

clearFilter?.addEventListener("click", () => {
  filterInput.value = "";
  renderSkills("");
});

renderSkills();

document.querySelectorAll(".reveal").forEach((section, i) => {
  section.style.animationDelay = `${i * 0.09}s`;
});
