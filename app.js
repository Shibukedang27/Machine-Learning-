const storageKeys = {
  profile: "retroos-calc-profile",
  history: "retroos-calc-history",
  usage: "retroos-calc-usage"
};

const modes = {
  basic: {
    title: "Quick arithmetic",
    desc: "Best for daily numbers, percentages, totals, and fast checks."
  },
  algebra: {
    title: "Algebra trainer",
    desc: "Great for equations, factors, powers, and step-by-step school work."
  },
  finance: {
    title: "Money calculator",
    desc: "Useful for profit, discount, EMI-style estimates, budgets, and tax checks."
  },
  science: {
    title: "Science helper",
    desc: "Handles powers, roots, ratios, and formula-style calculation practice."
  }
};

const input = document.querySelector("#expression");
const result = document.querySelector("#result");
const stepsList = document.querySelector("#steps-list");
const historyList = document.querySelector("#history-list");
const focusTitle = document.querySelector("#focus-title");
const focusDesc = document.querySelector("#focus-desc");
const ageInput = document.querySelector("#age-input");
const professionInput = document.querySelector("#profession-input");
const saveProfile = document.querySelector("#save-profile");
const intentPill = document.querySelector("#intent-pill");
const examToggle = document.querySelector("#exam-toggle");
const imageInput = document.querySelector("#image-input");
const imageName = document.querySelector("#image-name");
const voiceButton = document.querySelector("#voice-button");
const voiceLog = document.querySelector("#voice-log");

let activeMode = "basic";
let examMode = false;
let recognition;
let listening = false;

function readJson(key, fallback) {
  try {
    return JSON.parse(localStorage.getItem(key)) || fallback;
  } catch {
    return fallback;
  }
}

function writeJson(key, value) {
  localStorage.setItem(key, JSON.stringify(value));
}

function getProfile() {
  return readJson(storageKeys.profile, { age: 16, profession: "student" });
}

function getHistory() {
  return readJson(storageKeys.history, []);
}

function getUsage() {
  return readJson(storageKeys.usage, { basic: 0, algebra: 0, finance: 0, science: 0 });
}

function inferMode(expression = input.value) {
  const profile = getProfile();
  const text = expression.toLowerCase();
  const usage = getUsage();
  const weightedMode = Object.entries(usage).sort((a, b) => b[1] - a[1])[0]?.[0] || "basic";

  if (/[a-z]/.test(text) || text.includes("=") || text.includes("^")) return "algebra";
  if (/%|discount|profit|emi|tax|interest|budget/.test(text)) return "finance";
  if (/sqrt|sin|cos|tan|log|pi|root/.test(text)) return "science";
  if (profile.profession === "engineer") return "science";
  if (profile.profession === "business") return "finance";
  if (profile.profession === "student" && Number(profile.age) <= 18) return "algebra";
  return weightedMode;
}

function setMode(mode) {
  activeMode = mode;
  document.querySelectorAll(".mode").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });
  focusTitle.textContent = modes[mode].title;
  focusDesc.textContent = modes[mode].desc;
  intentPill.textContent = `Intent: ${mode}`;
}

function saveProfileData() {
  const age = Math.max(5, Math.min(100, Number(ageInput.value) || 16));
  const profession = professionInput.value;
  writeJson(storageKeys.profile, { age, profession });
  setMode(inferMode());
  generateSteps(input.value, result.value || result.textContent);
}

function sanitizeExpression(raw) {
  return raw
    .replaceAll("×", "*")
    .replaceAll("÷", "/")
    .replace(/\bplus\b/g, "+")
    .replace(/\bminus\b/g, "-")
    .replace(/\btimes\b|\bmultiplied by\b/g, "*")
    .replace(/\bdivided by\b|\bover\b/g, "/")
    .replace(/\bpercent\b/g, "%")
    .replace(/\bcalculate\b|\bwhat is\b|\bsolve\b/g, "")
    .trim();
}

function toJsExpression(expression) {
  return sanitizeExpression(expression)
    .replace(/(\d+(?:\.\d+)?)%/g, "($1/100)")
    .replace(/\^/g, "**")
    .replace(/\bpi\b/gi, "Math.PI")
    .replace(/\bsqrt\(/gi, "Math.sqrt(")
    .replace(/\bsin\(/gi, "Math.sin(")
    .replace(/\bcos\(/gi, "Math.cos(")
    .replace(/\btan\(/gi, "Math.tan(")
    .replace(/\blog\(/gi, "Math.log10(");
}

function isSafeExpression(expression) {
  return /^[\d+\-*/().%\s*MatPIQRSqrtcoginla]+$/.test(expression);
}

function formatNumber(value) {
  if (!Number.isFinite(value)) return "Cannot solve";
  return Number.parseFloat(value.toFixed(10)).toString();
}

function parseLinearSide(side) {
  const normalized = side.replace(/\s+/g, "").replace(/-/g, "+-");
  return normalized.split("+").filter(Boolean).reduce((total, term) => {
    if (term.includes("x")) {
      const coefficientText = term.replace("x", "");
      const coefficient = coefficientText === "" ? 1 : coefficientText === "-" ? -1 : Number(coefficientText);
      return { x: total.x + coefficient, n: total.n };
    }
    return { x: total.x, n: total.n + Number(term) };
  }, { x: 0, n: 0 });
}

function solveLinearEquation(expression) {
  if (!/^[\d+\-xX=.\s]+$/.test(expression) || !expression.includes("=")) return null;
  const [left, right] = expression.toLowerCase().split("=");
  if (!left || !right) return null;

  const leftSide = parseLinearSide(left);
  const rightSide = parseLinearSide(right);
  if ([leftSide.x, leftSide.n, rightSide.x, rightSide.n].some((value) => Number.isNaN(value))) return null;

  const xCoefficient = leftSide.x - rightSide.x;
  const constant = rightSide.n - leftSide.n;
  if (xCoefficient === 0) return null;

  return {
    answer: `x = ${formatNumber(constant / xCoefficient)}`,
    xCoefficient,
    constant
  };
}

function calculate(shouldStore = true) {
  const expression = input.value.trim();
  if (!expression) {
    result.textContent = "0";
    generateSteps("", "0");
    return;
  }

  const inferred = inferMode(expression);
  setMode(inferred);

  const linear = solveLinearEquation(expression);
  if (linear) {
    result.textContent = linear.answer;
    generateSteps(expression, linear.answer, linear);
    if (shouldStore) storeCalculation(expression, linear.answer, "algebra");
    return;
  }

  const jsExpression = toJsExpression(expression);

  if (!isSafeExpression(jsExpression) || /[a-z]/i.test(jsExpression.replace(/Math|PI|sqrt|sin|cos|tan|log/g, ""))) {
    result.textContent = "Need equation solver";
    generateSteps(expression, result.textContent);
    return;
  }

  try {
    const value = Function(`"use strict"; return (${jsExpression});`)();
    const answer = formatNumber(value);
    result.textContent = answer;
    generateSteps(expression, answer);
    if (shouldStore) storeCalculation(expression, answer, inferred);
  } catch {
    result.textContent = "Check input";
    generateSteps(expression, result.textContent);
  }
}

function storeCalculation(expression, answer, mode) {
  const history = getHistory();
  history.unshift({
    expression,
    answer,
    mode,
    at: new Date().toLocaleString()
  });
  writeJson(storageKeys.history, history.slice(0, 12));

  const usage = getUsage();
  usage[mode] = (usage[mode] || 0) + 1;
  writeJson(storageKeys.usage, usage);
  renderHistory();
}

function generateSteps(expression, answer, linear = null) {
  stepsList.innerHTML = "";
  const mode = activeMode;
  const clean = expression.trim();
  const steps = [];

  if (!clean) {
    steps.push("Enter a calculation or use voice/image input to begin.");
  } else if (linear) {
    steps.push("Detected a one-variable linear equation.");
    steps.push(`Move x terms to one side and constants to the other: ${formatNumber(linear.xCoefficient)}x = ${formatNumber(linear.constant)}.`);
    steps.push(`Divide both sides by ${formatNumber(linear.xCoefficient)}.`);
    steps.push(`Final answer: ${answer}.`);
  } else if (answer === "Need equation solver") {
    steps.push("Detected variables or an equation, so algebra mode is recommended.");
    steps.push("Rewrite the expression as a clean equation, such as 2x + 5 = 15.");
    steps.push("Isolate the variable by applying the same operation on both sides.");
  } else if (answer === "Check input") {
    steps.push("The expression has mismatched symbols or unsupported text.");
    steps.push("Use numbers, brackets, decimal points, and operators + - * / ^.");
  } else {
    steps.push(`Identify the task as ${mode} based on your profile and input pattern.`);
    steps.push(`Simplify the expression: ${clean}.`);
    steps.push(`Final answer: ${answer}.`);
  }

  if (examMode && clean) {
    steps.push("Exam scoring tip: write the formula, substitute values, show simplification, then box the final answer with units when needed.");
  }

  steps.forEach((step) => {
    const item = document.createElement("li");
    item.textContent = step;
    stepsList.append(item);
  });
}

function renderHistory() {
  const history = getHistory();
  historyList.innerHTML = "";

  if (history.length === 0) {
    historyList.innerHTML = '<p class="empty-history">No calculations stored yet. Your history stays in this browser on this device.</p>';
    return;
  }

  history.forEach((item) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "history-item";
    row.innerHTML = `<strong>${item.expression}</strong><span>= ${item.answer}</span>`;
    row.addEventListener("click", () => {
      input.value = item.expression;
      setMode(item.mode);
      calculate(false);
    });
    historyList.append(row);
  });
}

function insertText(text) {
  const start = input.selectionStart ?? input.value.length;
  const end = input.selectionEnd ?? input.value.length;
  input.value = `${input.value.slice(0, start)}${text}${input.value.slice(end)}`;
  input.focus();
  input.setSelectionRange(start + text.length, start + text.length);
  setMode(inferMode());
}

function setupVoice() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    voiceButton.disabled = true;
    voiceLog.textContent = "Voice input is not supported in this browser. You can still type or insert an image.";
    return;
  }

  recognition = new SpeechRecognition();
  recognition.lang = "en-IN";
  recognition.continuous = false;
  recognition.interimResults = false;
  recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    const expression = sanitizeExpression(transcript);
    voiceLog.textContent = `Heard: ${transcript}`;
    input.value = expression;
    calculate();
  };
  recognition.onend = () => {
    listening = false;
    voiceButton.classList.remove("listening");
    voiceButton.textContent = "Start Voice";
  };
}

document.querySelector(".keypad").addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button) return;

  if (button.dataset.insert) {
    insertText(button.dataset.insert);
    return;
  }

  if (button.dataset.key === "clear") {
    input.value = "";
    result.textContent = "0";
    generateSteps("", "0");
  }

  if (button.dataset.key === "back") {
    input.value = input.value.slice(0, -1);
    setMode(inferMode());
  }

  if (button.dataset.key === "equals") {
    calculate();
  }
});

document.querySelectorAll(".mode").forEach((button) => {
  button.addEventListener("click", () => setMode(button.dataset.mode));
});

input.addEventListener("input", () => setMode(inferMode()));
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter") calculate();
});

saveProfile.addEventListener("click", saveProfileData);

examToggle.addEventListener("click", () => {
  examMode = !examMode;
  examToggle.setAttribute("aria-pressed", String(examMode));
  generateSteps(input.value, result.textContent);
});

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (!file) return;
  imageName.textContent = file.name;
  input.value = "";
  result.textContent = "Image ready";
  setMode("algebra");
  stepsList.innerHTML = "";
  [
    "Image inserted locally. A production version would run on-device OCR before solving.",
    "Crop the equation clearly, then confirm the detected text in the input field.",
    "Exam mode can turn the solved expression into scoring steps."
  ].forEach((step) => {
    const item = document.createElement("li");
    item.textContent = step;
    stepsList.append(item);
  });
});

voiceButton.addEventListener("click", () => {
  if (!recognition) return;
  if (listening) {
    recognition.stop();
    return;
  }
  listening = true;
  voiceButton.classList.add("listening");
  voiceButton.textContent = "Listening";
  recognition.start();
});

function boot() {
  const profile = getProfile();
  ageInput.value = profile.age;
  professionInput.value = profile.profession;
  setMode(inferMode());
  generateSteps("", "0");
  renderHistory();
  setupVoice();
}

boot();
