#!/usr/bin/env python3
"""
=====================================================
HACKATHON TEMPLATE — Track 4
AI Project Manager Agent for Agile Teams
=====================================================
Starter template. Build on top of this.
DO NOT change class interfaces or output format.

You are provided:
  - A Jira-like dataset (CSV/JSON) of issues
  - Example API endpoints to fetch issues
  - Ground-truth story points and team ownership for a labeled subset

Required output format for scoring:
{
    "team_id": "MLX",
    "track": "track_4_pm_agent",
    "results": {
        "generated_tickets": [...],
        "story_points": {ticket_id: estimated_points},
        "team_assignments": {ticket_id: team_name},
        "blockers_detected": [...],
        "dependencies": [...],
        "daily_summary": "..."
    }
}
=====================================================
"""

import os
import json
import csv
import logging
import re
import requests
import joblib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG = {
    "team_id": "MLX",       # ← CHANGE THIS
    "llm_provider": "openai",
    "llm_model": "gpt-4o-mini",
    "api_base_url": "http://localhost:8000/api",  # Mock API provided by organizers
    "data_path": "./data/",
    "ticket-model-path": "./models/ticket_gen_model.pkl",
    "storypoint_model_path": "./models/story_point_model.pkl",
}


# ─────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────
class IssueType(Enum):
    EPIC = "epic"
    STORY = "story"
    BUG = "bug"
    TASK = "task"
    SUB_TASK = "sub_task"

class IssueStatus(Enum):
    BACKLOG = "backlog"
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    IN_REVIEW = "in_review"
    DONE = "done"
    BLOCKED = "blocked"

@dataclass
class Issue:
    issue_id: str
    title: str
    description: str
    issue_type: str           # epic, story, bug, task
    status: str               # backlog, todo, in_progress, in_review, done, blocked
    priority: str             # critical, high, medium, low
    assignee: Optional[str] = None
    team: Optional[str] = None
    story_points: Optional[int] = None
    parent_id: Optional[str] = None
    labels: List[str] = field(default_factory=list)
    comments: List[Dict] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # list of issue_ids
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

@dataclass
class GeneratedTicket:
    ticket_id: str
    title: str
    description: str
    issue_type: str
    acceptance_criteria: List[str]
    estimated_story_points: int
    assigned_team: str
    priority: str
    labels: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

@dataclass
class BlockerAlert:
    issue_id: str
    blocker_type: str       # dependency, stale, overdue, resource
    description: str
    severity: str           # critical, high, medium, low
    recommended_action: str

@dataclass
class DailySummary:
    date: str
    total_issues: int
    in_progress: int
    blocked: int
    completed_today: int
    at_risk: List[Dict]
    blockers: List[BlockerAlert]
    key_updates: List[str]
    action_items: List[str]


# ─────────────────────────────────────────────────────
# DATA INGESTION — Read backlog from API/CSV/JSON
# ─────────────────────────────────────────────────────
class BacklogReader:
    """Reads issues from multiple data sources."""

    @staticmethod
    def from_csv(filepath: str) -> List[Issue]:
        """Read issues from CSV file."""
        issues = []
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                issue = Issue(
                    issue_id=f"GH-{idx+1}",
                    title=row.get("title", ""),
                    description=row.get("user_story", ""),
                    issue_type="story",
                    status="backlog",
                    priority="medium",
                    assignee=None,
                    team=None,
                    story_points=int(row["point"]) if row.get("point") else None,
                    parent_id=None,
                    labels=[row.get("project")] if row.get("project") else [],
                    dependencies=[],
                    created_at=None,
                    updated_at=None,
                )
                issues.append(issue)
        return issues

    @staticmethod
    def from_json(filepath: str) -> List[Issue]:
        """Read issues from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return [Issue(**item) for item in data]

    @staticmethod
    def from_api(base_url: str, endpoint: str = "/issues") -> List[Issue]:
        """Read issues from REST API."""
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            response.raise_for_status()
            data = response.json()
            return [Issue(**item) for item in data]
        except Exception as e:
            logger.error(f"API fetch failed: {e}")
            return []


# ─────────────────────────────────────────────────────
# ABSTRACT INTERFACES
# ─────────────────────────────────────────────────────
class TicketGenerator(ABC):
    @abstractmethod
    def generate_tickets(self, feature_description: str,
                         existing_backlog: List[Issue]) -> List[GeneratedTicket]:
        """Break down a feature into implementable tickets."""
        pass

class StoryPointEstimator(ABC):
    @abstractmethod
    def estimate(self, ticket: GeneratedTicket, historical_data: List[Issue]) -> int:
        """Estimate story points for a ticket."""
        pass

class TeamAssigner(ABC):
    @abstractmethod
    def assign_team(self, ticket: GeneratedTicket, teams: Dict[str, List[str]]) -> str:
        """Assign a ticket to the most appropriate team."""
        pass

class BlockerDetector(ABC):
    @abstractmethod
    def detect_blockers(self, issues: List[Issue]) -> List[BlockerAlert]:
        """Analyze issues to detect blockers and at-risk items."""
        pass

class SummaryGenerator(ABC):
    @abstractmethod
    def generate_daily_summary(self, issues: List[Issue],
                               blockers: List[BlockerAlert]) -> DailySummary:
        """Generate a daily leadership summary."""
        pass


# ─────────────────────────────────────────────────────
# REFERENCE IMPLEMENTATIONS
# ─────────────────────────────────────────────────────

class TicketTopicModelTrainer:
    """Trains and saves TF-IDF + KMeans topic model."""

    def __init__(self, n_clusters=8):
        self.n_clusters = n_clusters

    def train_and_save(self, issues: List[Issue], save_path: str):

        texts = [
            f"{i.title} {i.description}"
            for i in issues
            if i.title or i.description
        ]

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000
        )

        X = vectorizer.fit_transform(texts)

        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10
        )

        kmeans.fit(X)

        terms = vectorizer.get_feature_names_out()
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

        cluster_keywords = {}

        for cluster_idx in range(self.n_clusters):
            top_terms = [terms[ind] for ind in order_centroids[cluster_idx, :10]]
            cluster_keywords[cluster_idx] = top_terms

        model_package = {
            "vectorizer": vectorizer,
            "kmeans": kmeans,
            "cluster_keywords": cluster_keywords
        }

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(model_package, save_path)

        print("Ticket topic model trained and saved.")

class TicketTopicModel:
    """Loads and uses pre-trained topic model."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}"
            )

    def predict_cluster(self, text: str) -> int:
        X = self.model["vectorizer"].transform([text])
        cluster = self.model["kmeans"].predict(X)[0]
        return cluster

    def predict_clusters_bulk(self, texts: List[str]) -> List[int]:
        X = self.model["vectorizer"].transform(texts)
        return self.model["kmeans"].predict(X)

    def get_vectorizer(self):
        return self.model["vectorizer"]

    def get_kmeans(self):
        return self.model["kmeans"]

    def get_cluster_keywords(self, cluster_id: int):
        return self.model["cluster_keywords"].get(cluster_id, [])

class LLMTicketGenerator(TicketGenerator):

    def __init__(self):
        model_path = CONFIG["ticket-model-path"]

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Ticket topic model not found at {model_path}. "
                "Ensure load_backlog() is called before generating tickets."
            )

        self.topic_model = TicketTopicModel(model_path)

    def generate_tickets(self, feature_description: str, existing_backlog: List[Issue]) -> List[GeneratedTicket]:

        # 1️⃣ Predict cluster of feature
        cluster = self.topic_model.predict_cluster(feature_description)

        # 2️⃣ Prepare backlog texts
        backlog_texts = [
            f"{i.title} {i.description}"
            for i in existing_backlog
            if i.title or i.description
        ]

        # 3️⃣ Predict clusters for backlog
        backlog_clusters = self.topic_model.predict_clusters_bulk(backlog_texts)

        # 4️⃣ Filter similar issues
        similar_issues = [
            issue for issue, c in zip(existing_backlog, backlog_clusters)
            if c == cluster
        ]

        # Fallback if cluster empty
        if not similar_issues:
            similar_issues = existing_backlog[:5]

        # 5️⃣ Extract common action verbs from titles
        verbs = []
        for issue in similar_issues:
            words = issue.title.split()
            if words:
                verbs.append(words[0].lower())

        # Most common verbs
        from collections import Counter
        common_verbs = [v for v, _ in Counter(verbs).most_common(3)]

        if not common_verbs:
            common_verbs = ["Design", "Implement", "Test"]

        # 6️⃣ Generate tickets dynamically
        generated = []
        counter = 1

        for verb in common_verbs:
            ticket = GeneratedTicket(
                ticket_id=f"GEN-{counter:03d}",
                title=f"{verb.capitalize()} {feature_description}",
                description=f"{verb.capitalize()} tasks related to {feature_description}",
                issue_type="story",
                acceptance_criteria=[
                    "Feature component completed",
                    "Unit tests added",
                    "Integration verified"
                ],
                estimated_story_points=0,
                assigned_team="",
                priority="medium",
                labels=[],
                dependencies=[]
            )
            generated.append(ticket)
            counter += 1

        # 7️⃣ Auto dependency chaining
        for i in range(1, len(generated)):
            generated[i].dependencies.append(generated[i-1].ticket_id)

        return generated

class StoryPointModelTrainer:
    """Trains and saves TF-IDF + RandomForest model for story point estimation."""

    FIBONACCI = [1, 2, 3, 5, 8, 13, 21]

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            min_df=3,
            max_df=0.9,
            sublinear_tf=True
        )
        self.model = RandomForestRegressor(
            n_estimators=100,
            n_jobs=-1,
            random_state=42
        )

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""

        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def train_and_save(self, issues: List[Issue], save_path: str):

        texts = []
        labels = []

        for issue in issues:
            if issue.story_points is None:
                continue

            combined = f"{issue.title} {issue.description}"
            cleaned = self._clean_text(combined)

            if len(cleaned.split()) < 3:
                continue

            texts.append(cleaned)
            labels.append(issue.story_points)
            
        if not texts:
            print("Warning: No labeled story points found. Skipping story point model training.")
            return
        X = self.vectorizer.fit_transform(texts)
        y = np.array(labels)

        self.model.fit(X, y)

        model_package = {
            "vectorizer": self.vectorizer,
            "model": self.model
        }

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(model_package, save_path)

        print("Story point model trained and saved.")

class StoryPointModel:
    """Loads and performs inference using trained story point model."""

    FIBONACCI = [1, 2, 3, 5, 8, 13, 21]

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Story point model not found at {model_path}"
            )

        self.model_package = joblib.load(model_path)

    def _snap_to_fibonacci(self, value: int) -> int:
        return min(self.FIBONACCI, key=lambda x: abs(x - value))

    def predict(self, ticket: GeneratedTicket) -> int:
        text = f"{ticket.title} {ticket.description}"
        X = self.model_package["vectorizer"].transform([text])
        raw_prediction = self.model_package["model"].predict(X)[0]
        return self._snap_to_fibonacci(int(round(raw_prediction)))

class LLMStoryPointEstimator(StoryPointEstimator):

    def __init__(self):
        self.model_path = CONFIG["storypoint_model_path"]
        self.model = StoryPointModel(self.model_path)

    def estimate(self, ticket: GeneratedTicket,
                 historical_data: List[Issue]) -> int:

        return self.model.predict(ticket)


class RuleBasedTeamAssigner(TeamAssigner):
    """Assign tickets to teams based on labels and skills."""

    TEAM_SKILLS = {
        "frontend": ["UI", "UX", "React", "CSS", "frontend", "design"],
        "backend": ["API", "database", "server", "backend", "microservice"],
        "ml_team": ["ML", "model", "training", "data", "pipeline", "AI"],
        "devops": ["deployment", "CI/CD", "infrastructure", "monitoring", "cloud"],
        "mobile": ["iOS", "Android", "mobile", "app"],
    }

    def assign_team(self, ticket: GeneratedTicket, teams: Dict[str, List[str]] = None) -> str:
        # ──────────────────────────────────────
        # TODO: ENHANCE with LLM/ML-based assignment
        # ──────────────────────────────────────
        skill_map = teams or self.TEAM_SKILLS
        text = f"{ticket.title} {ticket.description} {' '.join(ticket.labels)}".lower()
        scores = {}
        for team, keywords in skill_map.items():
            scores[team] = sum(1 for kw in keywords if kw.lower() in text)
        if max(scores.values()) == 0:
            return "backend"  # default
        return max(scores, key=scores.get)


class AnalyticsBlockerDetector(BlockerDetector):
    """Detect blockers by analyzing issue states and dependencies."""

    def detect_blockers(self, issues: List[Issue]) -> List[BlockerAlert]:
        blockers = []
        issue_map = {i.issue_id: i for i in issues}

        for issue in issues:
            # Check 1: Explicitly blocked status
            if issue.status == "blocked":
                blockers.append(BlockerAlert(
                    issue_id=issue.issue_id,
                    blocker_type="status",
                    description=f"{issue.title} is marked as blocked",
                    severity="high",
                    recommended_action="Identify and resolve blocking dependency"
                ))

            # Check 2: Dependency not done
            for dep_id in issue.dependencies:
                dep = issue_map.get(dep_id)
                if dep and dep.status != "done":
                    blockers.append(BlockerAlert(
                        issue_id=issue.issue_id,
                        blocker_type="dependency",
                        description=f"{issue.title} blocked by {dep.title} (status: {dep.status})",
                        severity="high" if issue.priority in ["critical", "high"] else "medium",
                        recommended_action=f"Prioritize completing {dep_id}"
                    ))

            # Check 3: Stale items (in_progress too long)
            if issue.status == "in_progress" and issue.updated_at:
                try:
                    last_update = datetime.fromisoformat(issue.updated_at)
                    days_stale = (datetime.now() - last_update).days
                    if days_stale > 5:
                        blockers.append(BlockerAlert(
                            issue_id=issue.issue_id,
                            blocker_type="stale",
                            description=f"{issue.title} unchanged for {days_stale} days",
                            severity="medium",
                            recommended_action="Check with assignee for status update"
                        ))
                except (ValueError, TypeError):
                    pass

        return blockers


class LLMSummaryGenerator(SummaryGenerator):
    """Generate daily summaries using LLM."""
    def generate_daily_summary(self, issues: List[Issue],
                           blockers: List[BlockerAlert]) -> DailySummary:

        status_counts = {}
        for issue in issues:
            status_counts[issue.status] = status_counts.get(issue.status, 0) + 1

        at_risk = [{"issue_id": b.issue_id, "reason": b.description}
                for b in blockers if b.severity in ["critical", "high"]]

        summary_text = f"""Daily Engineering Summary:

        Total Issues: {len(issues)}
        In Progress: {status_counts.get("in_progress", 0)}
        Blocked: {status_counts.get("blocked", 0)}

        High Risk Items: {len(at_risk)}

        Immediate Focus:
        - Resolve dependency blockers
        - Follow up on stale tickets
        - Prioritize critical issues
        """

        return DailySummary(
            date=datetime.now().strftime("%Y-%m-%d"),
            total_issues=len(issues),
            in_progress=status_counts.get("in_progress", 0),
            blocked=status_counts.get("blocked", 0),
            completed_today=0,
            at_risk=at_risk,
            blockers=blockers,
            key_updates=[summary_text],
            action_items=[
                "Resolve dependency blockers",
                "Review stale tickets",
                "Prioritize high severity issues"
            ],
        )
    

# ─────────────────────────────────────────────────────
# MAIN PM AGENT
# ─────────────────────────────────────────────────────
class PMAgent:

    def __init__(self):
        self.reader = BacklogReader()
        self.ticket_gen = LLMSummaryGenerator()
        self.estimator = LLMStoryPointEstimator()
        self.assigner = RuleBasedTeamAssigner()
        self.blocker_det = AnalyticsBlockerDetector()
        self.summary_gen = LLMSummaryGenerator()
        self.backlog: List[Issue] = []
        logger.info("PM Agent initialized")

    def load_backlog(self, source: str, source_type: str = "csv") -> int:
        """Load backlog from data source. Returns count of issues loaded."""

        if source_type == "csv":
            self.backlog = self.reader.from_csv(source)
        elif source_type == "json":
            self.backlog = self.reader.from_json(source)
        elif source_type == "api":
            self.backlog = self.reader.from_api(source)

        logger.info(f"Loaded {len(self.backlog)} issues from {source_type}")

        # ───────────── RETRAIN MODELS ─────────────
        if not os.path.exists(CONFIG["ticket-model-path"]):
            logger.info("Ticket topic model not found. Training new model...")
            trainer = TicketTopicModelTrainer()
            trainer.train_and_save(self.backlog, CONFIG["ticket-model-path"])
            CONFIG["retrain_ticket_model"] = False

        if not os.path.exists(CONFIG["storypoint_model_path"]):
            logger.info("Story point model not found. Training new model...")
            sp_trainer = StoryPointModelTrainer()
            sp_trainer.train_and_save(self.backlog, CONFIG["storypoint_model_path"])
            CONFIG["retrain_storypoint_model"] = False

        # ───────────── INITIALIZE MODELS SAFELY ─────────────
        self.ticket_gen = LLMTicketGenerator()
        self.estimator = LLMStoryPointEstimator()

        return len(self.backlog)

    def force_retrain_models(self):
        """Force retraining of both models."""
        logger.info("Force retraining models...")

        trainer = TicketTopicModelTrainer()
        trainer.train_and_save(self.backlog, CONFIG["ticket-model-path"])

        sp_trainer = StoryPointModelTrainer()
        sp_trainer.train_and_save(self.backlog, CONFIG["storypoint_model_path"])

        # Reinitialize models after retraining
        self.ticket_gen = LLMTicketGenerator()
        self.estimator = LLMStoryPointEstimator()

        logger.info("Models retrained successfully.")

    def break_down_feature(self, feature_description: str) -> List[GeneratedTicket]:
        """Generate tickets from a high-level feature description."""
        tickets = self.ticket_gen.generate_tickets(feature_description, self.backlog)
        for ticket in tickets:
            ticket.estimated_story_points = self.estimator.estimate(ticket, self.backlog)
            ticket.assigned_team = self.assigner.assign_team(ticket)
        return tickets

    def detect_blockers(self) -> List[BlockerAlert]:
        """Scan backlog for blockers and at-risk items."""
        return self.blocker_det.detect_blockers(self.backlog)

    def generate_summary(self) -> DailySummary:
        """Generate daily leadership summary."""
        blockers = self.detect_blockers()
        return self.summary_gen.generate_daily_summary(self.backlog, blockers)

    def export_results(self, tickets: List[GeneratedTicket] = None, blockers: List[BlockerAlert] = None) -> Dict:
        summary = self.generate_summary()

        return {
            "team_id": CONFIG["team_id"],
            "track": "track_4_pm_agent",
            "results": {
                "generated_tickets": [asdict(t) for t in (tickets or [])],
                "story_points": {
                    t.ticket_id: t.estimated_story_points
                    for t in (tickets or [])
                },
                "team_assignments": {
                    t.ticket_id: t.assigned_team
                    for t in (tickets or [])
                },
                "blockers_detected": [asdict(b) for b in (blockers or [])],
                "dependencies": [
                    {
                        "ticket_id": t.ticket_id,
                        "depends_on": t.dependencies
                    }
                    for t in (tickets or [])
                    if t.dependencies
                ],
                "daily_summary": json.dumps(asdict(summary))
            }
        }


# ─────────────────────────────────────────────────────
# SAMPLE DATA
# ─────────────────────────────────────────────────────
SAMPLE_ISSUES = [
    Issue("ISS-001", "User login page", "Implement OAuth2 login flow", "story", "done", "high",
          team="frontend", story_points=5, labels=["auth", "UI"]),
    Issue("ISS-002", "Payment API integration", "Integrate Razorpay payment gateway", "story", "in_progress", "critical",
          team="backend", story_points=8, labels=["payments", "API"],
          dependencies=["ISS-003"], updated_at="2026-02-15"),
    Issue("ISS-003", "Database schema for orders", "Design order tables", "task", "in_progress", "high",
          team="backend", story_points=3, labels=["database"],
          updated_at="2026-02-10"),
    Issue("ISS-004", "Mobile push notifications", "Implement FCM push", "story", "blocked", "medium",
          team="mobile", story_points=5, labels=["mobile", "notifications"],
          dependencies=["ISS-005"]),
    Issue("ISS-005", "Notification service API", "Build notification microservice", "story", "todo", "high",
          team="backend", story_points=8, labels=["API", "microservice"]),
]

if __name__ == "__main__":
    agent = PMAgent()
    agent.load_backlog(source="data.csv", source_type="csv")

    agent.force_retrain_models()  # Force retrain models on loaded backlog
    
    print("=" * 50)
    print("AI Project Manager Agent — Demo")
    print("=" * 50)

    # Blocker detection works out of the box
    print("\nBlocker Detection (functional demo):")
    blockers = agent.detect_blockers()
    for b in blockers:
        print(f"  ⚠️  [{b.severity.upper()}] {b.issue_id}: {b.description}")
        print(f"     → {b.recommended_action}")

    # Daily summary works out of the box
    print("\nDaily Summary:")
    summary = agent.generate_summary()
    print(f"  Total Issues: {summary.total_issues}")
    print(f"  In Progress: {summary.in_progress}")
    print(f"  Blocked: {summary.blocked}")
    print(f"  At-Risk Items: {len(summary.at_risk)}")

    # Team assignment works out of the box
    print("\nTeam Assignment Demo:")
    test_ticket = GeneratedTicket(
        ticket_id="NEW-001", title="Build ML recommendation engine",
        description="Train collaborative filtering model for product recommendations",
        issue_type="story", acceptance_criteria=["Model accuracy > 80%"],
        estimated_story_points=8, assigned_team="", priority="high",
        labels=["ML", "model", "training"]
    )
    team = agent.assigner.assign_team(test_ticket)
    print(f"  Ticket: {test_ticket.title}")
    print(f"  Assigned to: {team}")

    # Export
    output = agent.export_results(blockers=blockers)
    print(f"\nExport format preview: {json.dumps(output, indent=2)[:500]}...")