// src/api/client.js
import axios from "axios";

const API_BASE = "http://localhost:8000";

export async function askAssistant(question, chatHistory = []) {
  const response = await axios.post(`${API_BASE}/api/chat`, {
    question,
    chat_history: chatHistory,
  });
  return response.data; // { final_answer, routing_reason, trace }
}

export async function getAlerts() {
  const response = await axios.get(`${API_BASE}/api/alerts`);
  return response.data; // array of row objects
}

export async function getMonthlyTrend() {
  const response = await axios.get(`${API_BASE}/api/monthly-trend`);
  return response.data; // array of row objects
}
