# Phase 10: Agent Workflow Orchestration

## Goal

Add an orchestrated AI Agent workflow to DataAgent Analyst.

This phase uses LangGraph to coordinate existing tools such as dataset profiling, EDA, visualization recommendation, ML training, report generation, and AI insight generation.

## Backend Deliverables

- Agent schema models
- Agent workflow service
- LangGraph StateGraph workflow
- Agent API route
- Workflow steps and outputs
- End-to-end workflow tests

## Frontend Deliverables

- Agent Workflow Lab
- Agent goal input
- Workflow option checkboxes
- Run agent workflow button
- Workflow timeline
- Final outputs JSON panel

## Agent Workflow

```text
Planner
→ Data Profiler
→ EDA Runner
→ Visualization Recommender
→ ML Trainer
→ Report Writer
→ AI Insight Generator
→ Finalizer
