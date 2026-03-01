# netrikhackathon1-mlx
AI Project Manager Agent

NETRIK Hackathon 2026 — Track 2 Submission
Team ID: MLX
Track: AI Project Manager Agent for Agile Teams

1. Overview

This project implements an AI-based Project Management Agent designed to support Agile development teams in backlog management, effort estimation, dependency tracking, and executive reporting.

The solution extends the official hackathon template while preserving all required class interfaces and output structures. The system is deterministic, modular, and compatible with the evaluation harness.

The agent provides:

Structured backlog ingestion

Feature decomposition into technical sub-tasks

Machine learning–based story point estimation

Skill-based team assignment

Cross-ticket dependency and blocker detection

Executive-level daily sprint summaries

Standardized JSON export compliant with evaluation schema

2. Functional Components
2.1 Backlog Ingestion

Supports the following input sources:

CSV files

JSON files

REST API endpoints

Backlog items are parsed into strongly-typed Issue dataclass objects to maintain schema integrity and enable structured analysis.

2.2 Ticket Decomposition

High-level feature descriptions are decomposed into:

Architecture design ticket

Backend implementation ticket

Frontend implementation ticket

Each generated ticket includes:

Acceptance criteria

Logical dependencies (Finish-to-Start sequencing)

Estimated story points

Assigned team

Priority

2.3 Story Point Estimation

The system uses a machine learning pipeline consisting of:

TF-IDF text vectorization

RandomForestRegressor

Fibonacci scale snapping (1, 2, 3, 5, 8, 13, 21)

Model behavior:

Automatically trains if model files are missing

Loads pre-trained models if available

Supports explicit manual retraining

This ensures reproducibility and deterministic behavior across executions.

2.4 Resource Assignment

Tickets are assigned using a rule-based skill matching engine.

Supported teams:

frontend

backend

ml_team

devops

mobile

Assignment logic evaluates:

Ticket title

Description

Labels

Keyword-based skill scoring

2.5 Blocker and Risk Detection

The system detects:

Explicitly blocked tickets

Cross-ticket dependency conflicts

High-priority dependency risks

Stale in-progress tickets (no updates for more than 5 days)

Each blocker includes:

Severity classification

Recommended corrective action

2.6 Executive Daily Summary

The daily summary includes:

Overall sprint progress overview

Active blockers

Identified risks

Immediate action priorities

The summary is exported in structured JSON format for automated evaluation.

3. Export Format Compliance

The agent strictly adheres to the required evaluation schema:

{
    "team_id": "MLX",
    "track": "track_4_pm_agent",
    "results": {
        "generated_tickets": [...],
        "story_points": {...},
        "team_assignments": {...},
        "blockers_detected": [...],
        "dependencies": [...],
        "daily_summary": "..."
    }
}

This guarantees compatibility with the hackathon’s automated scoring engine.

4. Architecture Overview

Core components:

PMAgent

BacklogReader

TicketTopicModelTrainer

StoryPointModelTrainer

LLMTicketGenerator

LLMStoryPointEstimator

RuleBasedTeamAssigner

AnalyticsBlockerDetector

LLMSummaryGenerator

Design principles:

Strict template interface preservation

Modular ML components

Deterministic training logic

Explicit retraining control

Clean object-oriented separation of concerns

5. Installation
5.1 Requirements

Python 3.9+

scikit-learn

numpy

joblib

streamlit

graphviz

requests

Install dependencies:

pip install -r requirements.txt
6. Running the Project
6.1 Backend (CLI Mode)
python mlx.py

This will:

Load the backlog dataset

Train models if missing

Perform blocker detection

Generate summary

Produce evaluation JSON output

6.2 Force Model Retraining

Manual retraining can be triggered programmatically:

agent.force_retrain_models()

This retrains:

Ticket topic clustering model

Story point estimation model

6.3 Streamlit Interface
streamlit run app.py

The web interface provides:

Feature decomposition input

Ticket visualization

Dependency graph

Blocker dashboard

Executive summary

JSON export preview

7. Project Structure
├── mlx.py               # Core AI PM Agent implementation
├── app.py               # Streamlit user interface
├── models/              # Saved ML models
├── data.csv             # Sample backlog dataset
├── requirements.txt
└── README.md
8. Reproducibility and Determinism

Fixed random seeds in ML models

File-based model existence checks

No runtime retrain flags

Deterministic export schema

Outputs remain consistent across repeated executions on the same dataset.

9. Assumptions

CSV dataset includes: title, user_story, point, project

Dates are in ISO 8601 format for stale detection

Fibonacci scale is used for story point normalization

Team skill mapping is static

10. Submission Information

NETRIK Hackathon 2026
Track 2 — AI Project Manager Agent
Team MLX
