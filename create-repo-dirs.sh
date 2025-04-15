#!/bin/bash

# Create top-level directories for each category
mkdir -p core-processing
mkdir -p architectural
mkdir -p efficiency
mkdir -p memory-management
mkdir -p safety
mkdir -p human-collaboration
mkdir -p orchestration
mkdir -p explainability
mkdir -p resilience

# Core Processing Patterns
mkdir -p core-processing/retrieval-augmented-generation
mkdir -p core-processing/chain-of-thought-prompting
mkdir -p core-processing/react
mkdir -p core-processing/reflection

# Architectural Patterns
mkdir -p architectural/router
mkdir -p architectural/planner
mkdir -p architectural/tool-use
mkdir -p architectural/multi-agent
mkdir -p architectural/hierarchical-task-decomposition

# Efficiency Patterns
mkdir -p efficiency/semantic-caching
mkdir -p efficiency/complexity-based-routing
mkdir -p efficiency/dynamic-prompt-engineering
mkdir -p efficiency/fallback-chains

# Memory Management Patterns
mkdir -p memory-management/episodic-memory
mkdir -p memory-management/declarative-knowledge-bases
mkdir -p memory-management/procedural-memory

# Safety Patterns
mkdir -p safety/input-filtering
mkdir -p safety/output-filtering
mkdir -p safety/constitutions-and-principles
mkdir -p safety/sandboxing
mkdir -p safety/tool-usage-permission-systems

# Human Collaboration Patterns
mkdir -p human-collaboration/confidence-based-human-escalation
mkdir -p human-collaboration/interactive-refinement
mkdir -p human-collaboration/feedback-collection-and-integration

# Orchestration Patterns
mkdir -p orchestration/workflow-management
mkdir -p orchestration/asynchronous-processing

# Explainability Patterns
mkdir -p explainability/process-transparency
mkdir -p explainability/decision-trail-recording
mkdir -p explainability/alternative-exploration

# Resilience Patterns
mkdir -p resilience/graceful-degradation
mkdir -p resilience/error-recovery-strategies
mkdir -p resilience/rate-limiting-and-throttling

echo "Directory structure for agentic AI design patterns created successfully."