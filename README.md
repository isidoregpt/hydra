
# 🧠 Hydra v2 — Multi-Model AI Orchestrator

Hydra v2 is a clean-room multi-model AI orchestration framework built for real-time collaborative problem solving across multiple LLM providers:

- 🤖 OpenAI GPT-4o (`gpt-4o`)
- 🤖 Anthropic Claude Opus 4 (`claude-opus-4-20250514`)
- 🤖 Google Gemini 2.5 Pro (`gemini-2.5-pro-preview-06-05`)

The models collaborate together to analyze, debate, and produce multi-perspective responses.

## ⚙ How It Works

- User enters a problem/task.
- Hydra orchestrates:
  - Gemini: initial analysis & breakdown
  - OpenAI: reasoning & recommendation
  - Claude: synthesis & expert plan

## 🚀 Deployment

1. Upload to Streamlit Cloud
2. Set `app.py` as the entrypoint
3. Enter API keys when prompted inside app

## 🔑 Required API Keys

- OpenAI: https://platform.openai.com/account/api-keys
- Anthropic: https://console.anthropic.com/settings/keys
- Google Gemini: https://aistudio.google.com/app/apikey

## 📂 Project Structure

- app.py - Streamlit entrypoint
- orchestrator.py - Multi-agent orchestration logic
- models/ - Individual agent wrappers
- config.py - Config placeholder for future expansion
- requirements.txt - Python dependencies

Hydra is extensible, simple, and fully serverless on Streamlit Cloud.
