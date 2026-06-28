# Phase 13: Project Hardening, Modular Frontend, and Portfolio Toolkit

## Goal

Improve reliability, maintainability, frontend structure, developer workflow, and portfolio presentation.

## Backend Deliverables

- Structured API error responses
- Request ID middleware
- Request processing time header
- Global HTTP exception handler
- Global validation exception handler
- Global unhandled exception handler
- Agent job detail bug fix
- Dataset validation before agent job creation
- Error handling tests

## Frontend Deliverables

- HTML partial architecture
- Workspace navigation
- Bootstrap module
- Dynamic partial loading
- Global toast notifications
- Frontend DOM contract checker

## Developer Tooling

- Makefile
- Project check script
- End-to-end smoke test
- Deterministic demo dataset generator
- Portfolio demo documentation
- Screenshot checklist

## Frontend Structure

```text
frontend/
├── index.html
├── partials/
│   ├── overview.html
│   ├── analysis.html
│   ├── machine-learning.html
│   └── intelligence.html
├── css/
│   ├── base.css
│   ├── components.css
│   ├── dashboard.css
│   ├── layout.css
│   ├── navigation.css
│   └── toast.css
└── js/
    ├── bootstrap.js
    ├── app.js
    ├── api/
    ├── components/
    ├── ui/
    └── utils/
