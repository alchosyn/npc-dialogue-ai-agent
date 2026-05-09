# NPC Dialogue AI Agent

一个基于 LLM 的 NPC 对话 AI Agent，使用 Groq API 驱动，在 Google Colab 上运行。

## 项目简介

这是一个从零搭建的 AI Agent 项目。Agent 扮演一个叫"信噪"的赛博朋克角色，能进行多轮对话，并在需要时自主调用工具（查时间、做计算）。

## 功能

1.多轮角色扮演对话
2.工具调用（当前时间查询、数学计算）
3.完整的 Agent 循环：LLM 判断 → 调用工具 → 返回结果 → 生成回复
4.格式漂移清理（处理 Llama 模型的标签泄漏问题）

## 技术栈

**LLM**：llama-3.3-70b-versatile（通过 Groq API）
**开发环境**：Google Colab
**API 格式**：OpenAI 兼容

## 快速开始

1. 在 [Groq](https://console.groq.com/) 获取免费 API Key
2. 用 Google Colab 打开 `NPC_agent.ipynb`
3. 将 API Key 存入 Colab Secrets，变量名为 `GROQ_API_KEY`
4. 运行 cell，开始和信噪对话

## 项目状态

这是一个学习项目，持续开发中。
