// src/api/dashboard.js
import axios from "axios";

const API_BASE = "http://localhost:8000";

export async function getKpis(dataset) {
  const res = await axios.get(`${API_BASE}/api/dashboard/kpis`, { params: { dataset } });
  return res.data;
}

export async function getTrainingResults() {
  const res = await axios.get(`${API_BASE}/api/dashboard/training-results`);
  return res.data;
}

export async function getPredictionResults() {
  const res = await axios.get(`${API_BASE}/api/dashboard/prediction-results`);
  return res.data;
}

export function plotUrl(filename) {
  return `${API_BASE}/plots/${filename}`;
}
