const storageKey = "textclassify-dataset";

const starterDataset = [
  { text: "The product arrived quickly and the support team was helpful.", label: "Positive" },
  { text: "Amazing quality, fast delivery, and a smooth experience.", label: "Positive" },
  { text: "I loved the simple setup and friendly customer service.", label: "Positive" },
  { text: "This is reliable, polished, and worth recommending.", label: "Positive" },
  { text: "The app keeps crashing and nobody replied to my ticket.", label: "Negative" },
  { text: "Delivery was late and the item was damaged.", label: "Negative" },
  { text: "The instructions were confusing and the result was disappointing.", label: "Negative" },
  { text: "I am frustrated because the refund process is slow.", label: "Negative" },
  { text: "How do I reset my password for this account?", label: "Question" },
  { text: "Can you explain how the subscription billing works?", label: "Question" },
  { text: "What is the best way to upload a CSV file?", label: "Question" },
  { text: "Where can I find the invoice for last month?", label: "Question" }
];

const stopWords = new Set([
  "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
  "i", "in", "is", "it", "my", "of", "on", "or", "the", "this", "to",
  "was", "what", "where", "with", "you", "your"
]);

const elements = {
  textInput: document.querySelector("#text-input"),
  classifyButton: document.querySelector("#classify-button"),
  trainButton: document.querySelector("#train-button"),
  clearText: document.querySelector("#clear-text"),
  prediction: document.querySelector("#prediction"),
  confidenceBar: document.querySelector("#confidence-bar"),
  confidenceText: document.querySelector("#confidence-text"),
  sampleCount: document.querySelector("#sample-count"),
  classCount: document.querySelector("#class-count"),
  vocabCount: document.querySelector("#vocab-count"),
  accuracy: document.querySelector("#accuracy"),
  signals: document.querySelector("#signals"),
  datasetBody: document.querySelector("#dataset-body"),
  sampleForm: document.querySelector("#sample-form"),
  sampleText: document.querySelector("#sample-text"),
  sampleLabel: document.querySelector("#sample-label"),
  resetData: document.querySelector("#reset-data")
};

let dataset = loadDataset();
let model = train(dataset);

function loadDataset() {
  try {
    return JSON.parse(localStorage.getItem(storageKey)) || starterDataset;
  } catch {
    return starterDataset;
  }
}

function saveDataset() {
  localStorage.setItem(storageKey, JSON.stringify(dataset));
}

function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((token) => token.length > 1 && !stopWords.has(token));
}

function train(samples) {
  const labels = [...new Set(samples.map((sample) => sample.label))];
  const vocabulary = new Set();
  const classDocs = {};
  const classTokenCounts = {};
  const tokenCounts = {};

  labels.forEach((label) => {
    classDocs[label] = 0;
    classTokenCounts[label] = 0;
    tokenCounts[label] = {};
  });

  samples.forEach((sample) => {
    const tokens = tokenize(sample.text);
    classDocs[sample.label] += 1;
    tokens.forEach((token) => {
      vocabulary.add(token);
      tokenCounts[sample.label][token] = (tokenCounts[sample.label][token] || 0) + 1;
      classTokenCounts[sample.label] += 1;
    });
  });

  return {
    labels,
    vocabulary: [...vocabulary],
    classDocs,
    classTokenCounts,
    tokenCounts,
    totalDocs: samples.length
  };
}

function classify(text) {
  const tokens = tokenize(text);
  const vocabSize = Math.max(model.vocabulary.length, 1);
  const scores = model.labels.map((label) => {
    const prior = Math.log((model.classDocs[label] + 1) / (model.totalDocs + model.labels.length));
    const tokenScore = tokens.reduce((score, token) => {
      const count = model.tokenCounts[label][token] || 0;
      const probability = (count + 1) / (model.classTokenCounts[label] + vocabSize);
      return score + Math.log(probability);
    }, 0);
    return { label, score: prior + tokenScore };
  });

  const maxScore = Math.max(...scores.map((item) => item.score));
  const probabilities = scores.map((item) => ({
    label: item.label,
    probability: Math.exp(item.score - maxScore)
  }));
  const total = probabilities.reduce((sum, item) => sum + item.probability, 0) || 1;
  const normalized = probabilities
    .map((item) => ({ label: item.label, probability: item.probability / total }))
    .sort((a, b) => b.probability - a.probability);

  return {
    label: normalized[0]?.label || "Unknown",
    confidence: normalized[0]?.probability || 0,
    tokens
  };
}

function estimateAccuracy() {
  if (dataset.length < 4) return 1;

  let correct = 0;
  dataset.forEach((sample, index) => {
    const trainingFold = dataset.filter((_, sampleIndex) => sampleIndex !== index);
    const previousModel = model;
    model = train(trainingFold);
    if (classify(sample.text).label === sample.label) correct += 1;
    model = previousModel;
  });

  return correct / dataset.length;
}

function topSignals(tokens, label) {
  const uniqueTokens = [...new Set(tokens)];
  const vocabSize = Math.max(model.vocabulary.length, 1);
  return uniqueTokens
    .map((token) => {
      const count = model.tokenCounts[label]?.[token] || 0;
      const weight = (count + 1) / (model.classTokenCounts[label] + vocabSize);
      return { token, weight };
    })
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 5);
}

function updatePrediction() {
  const text = elements.textInput.value.trim();
  if (!text) {
    elements.prediction.textContent = "Waiting";
    elements.confidenceBar.style.width = "0%";
    elements.confidenceText.textContent = "Enter text to classify";
    elements.signals.innerHTML = "";
    return;
  }

  const prediction = classify(text);
  const confidence = Math.round(prediction.confidence * 100);
  elements.prediction.textContent = prediction.label;
  elements.confidenceBar.style.width = `${confidence}%`;
  elements.confidenceText.textContent = `Confidence ${confidence}%`;
  renderSignals(topSignals(prediction.tokens, prediction.label));
}

function renderSignals(signals) {
  elements.signals.innerHTML = "";
  if (signals.length === 0) {
    elements.signals.innerHTML = '<span class="signal">No strong tokens</span>';
    return;
  }

  signals.forEach((signal) => {
    const chip = document.createElement("span");
    chip.className = "signal";
    chip.textContent = signal.token;
    elements.signals.append(chip);
  });
}

function renderDataset() {
  elements.datasetBody.innerHTML = "";
  dataset.forEach((sample) => {
    const row = document.createElement("tr");
    row.innerHTML = `<td>${sample.text}</td><td>${sample.label}</td>`;
    elements.datasetBody.append(row);
  });
}

function renderMetrics() {
  elements.sampleCount.textContent = dataset.length;
  elements.classCount.textContent = model.labels.length;
  elements.vocabCount.textContent = model.vocabulary.length;
  elements.accuracy.textContent = `${Math.round(estimateAccuracy() * 100)}%`;
}

function retrain() {
  model = train(dataset);
  saveDataset();
  renderDataset();
  renderMetrics();
  updatePrediction();
}

elements.classifyButton.addEventListener("click", updatePrediction);
elements.trainButton.addEventListener("click", retrain);
elements.textInput.addEventListener("input", updatePrediction);

elements.clearText.addEventListener("click", () => {
  elements.textInput.value = "";
  updatePrediction();
});

elements.sampleForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = elements.sampleText.value.trim();
  if (!text) return;

  dataset.unshift({
    text,
    label: elements.sampleLabel.value
  });
  elements.sampleText.value = "";
  retrain();
});

elements.resetData.addEventListener("click", () => {
  dataset = [...starterDataset];
  retrain();
});

retrain();
