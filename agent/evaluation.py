"""Digest Evaluation Pipeline — mines approval/rejection history to measure pipeline quality."""

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import yaml

from config.settings import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_REVIEWS = 20         # Minimum total reviews for reliable aggregate metrics
MIN_BUCKET_SIZE = 5      # Minimum items per bucket/source for per-segment stats

RELEVANCE_BUCKETS = [(0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
EMBEDDING_BUCKETS = [(0.0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.6), (0.6, 1.01)]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ReviewedItem:
    """A parsed digest review item with extracted scores."""

    title: str
    source_name: str
    relevance_score: float
    embedding_score: float
    approved: bool
    created_at: str
    summary: str = ""
    url: str = ""


@dataclass
class EvalMetrics:
    """Aggregated evaluation metrics for the digest pipeline."""

    total_reviewed: int
    total_approved: int
    total_rejected: int
    approval_rate: float
    by_relevance_bucket: dict[str, dict] = field(default_factory=dict)
    by_embedding_bucket: dict[str, dict] = field(default_factory=dict)
    by_source: dict[str, dict] = field(default_factory=dict)
    optimal_relevance_threshold: float | None = None
    optimal_embedding_threshold: float | None = None
    weekly_trend: list[dict] = field(default_factory=list)
    data_sufficient: bool = False
    date_range: str = ""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_task_text(task_text: str) -> dict | None:
    """Extract structured fields from a [Digest Review] task text.

    Returns a dict with title, source_name, relevance_score, embedding_score,
    summary, and url. Returns None if the text cannot be parsed.
    """
    title_match = re.search(r"\[Digest Review\]\s*(.+)", task_text)
    if not title_match:
        return None

    title = title_match.group(1).strip()

    source_match = re.search(r"Source:\s*(.+)", task_text)
    source_name = source_match.group(1).strip() if source_match else ""

    relevance_match = re.search(r"Relevance:\s*([\d.]+)", task_text)
    relevance_score = float(relevance_match.group(1)) if relevance_match else 0.0

    embedding_match = re.search(r"Embedding:\s*([\d.]+)", task_text)
    embedding_score = float(embedding_match.group(1)) if embedding_match else 0.0

    summary_match = re.search(r"Summary:\s*(.+?)(?:\nURL:|\Z)", task_text, re.DOTALL)
    summary = summary_match.group(1).strip() if summary_match else ""

    url_match = re.search(r"URL:\s*(\S+)", task_text)
    url = url_match.group(1).strip() if url_match else ""

    return {
        "title": title,
        "source_name": source_name,
        "relevance_score": relevance_score,
        "embedding_score": embedding_score,
        "summary": summary,
        "url": url,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_reviewed_items(client=None, limit: int = 500) -> list[ReviewedItem]:
    """Fetch and parse reviewed digest items from SCMS."""
    if client is None:
        from scms.client import SCMSClient
        client = SCMSClient()

    rows = client.get_reviewed_digest_items(limit=limit)
    items: list[ReviewedItem] = []

    for row in rows:
        parsed = parse_task_text(row.get("task", ""))
        if parsed is None:
            logger.debug("Skipping unparseable task: %s", row.get("id", "?"))
            continue

        items.append(ReviewedItem(
            title=parsed["title"],
            source_name=parsed["source_name"],
            relevance_score=parsed["relevance_score"],
            embedding_score=parsed["embedding_score"],
            approved=row.get("status") == "completed",
            created_at=row.get("created_at", ""),
            summary=parsed["summary"],
            url=parsed["url"],
        ))

    return items


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def _bucket_label(low: float, high: float) -> str:
    """Format a bucket range as a label string."""
    if high > 1.0:
        return f"{low:.1f}+"
    return f"{low:.1f}-{high:.1f}"


def _bucket_items(items: list[ReviewedItem], buckets: list[tuple[float, float]],
                  score_attr: str) -> dict[str, dict]:
    """Group items into score buckets and compute approval rate per bucket."""
    result = {}
    for low, high in buckets:
        label = _bucket_label(low, high)
        in_bucket = [i for i in items if low <= getattr(i, score_attr) < high]
        total = len(in_bucket)
        approved = sum(1 for i in in_bucket if i.approved)
        result[label] = {
            "total": total,
            "approved": approved,
            "rate": approved / total if total > 0 else 0.0,
        }
    return result


def _compute_optimal_threshold(items: list[ReviewedItem], score_attr: str,
                               start: float = 0.0, stop: float = 1.0,
                               step: float = 0.05) -> float | None:
    """Find threshold that maximizes F1 score for approvals.

    At each candidate threshold:
    - Precision = approval rate of items >= threshold
    - Recall = fraction of all approved items that are >= threshold
    - F1 = harmonic mean of precision and recall
    """
    if len(items) < MIN_REVIEWS:
        return None

    total_approved = sum(1 for i in items if i.approved)
    if total_approved == 0:
        return None

    best_f1 = 0.0
    best_threshold = start

    t = start
    while t <= stop:
        above = [i for i in items if getattr(i, score_attr) >= t]
        if not above:
            t += step
            continue

        true_pos = sum(1 for i in above if i.approved)
        precision = true_pos / len(above) if above else 0.0
        recall = true_pos / total_approved if total_approved > 0 else 0.0

        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

        t += step

    return round(best_threshold, 2)


def _compute_weekly_trend(items: list[ReviewedItem]) -> list[dict]:
    """Group items by ISO week and compute approval rate per week."""
    weeks: dict[str, dict] = {}

    for item in items:
        try:
            dt = datetime.fromisoformat(item.created_at.replace("Z", "+00:00"))
            iso = dt.isocalendar()
            week_key = f"{iso.year}-W{iso.week:02d}"
        except (ValueError, AttributeError):
            continue

        if week_key not in weeks:
            weeks[week_key] = {"week": week_key, "total": 0, "approved": 0}
        weeks[week_key]["total"] += 1
        if item.approved:
            weeks[week_key]["approved"] += 1

    result = []
    for week_key in sorted(weeks.keys()):
        entry = weeks[week_key]
        entry["rate"] = entry["approved"] / entry["total"] if entry["total"] > 0 else 0.0
        result.append(entry)

    return result


def compute_metrics(items: list[ReviewedItem]) -> EvalMetrics:
    """Compute all evaluation metrics from a list of reviewed items."""
    total = len(items)
    approved = sum(1 for i in items if i.approved)
    rejected = total - approved

    # Date range
    dates = [i.created_at[:10] for i in items if i.created_at]
    date_range = f"{min(dates)} to {max(dates)}" if dates else "N/A"

    metrics = EvalMetrics(
        total_reviewed=total,
        total_approved=approved,
        total_rejected=rejected,
        approval_rate=approved / total if total > 0 else 0.0,
        data_sufficient=total >= MIN_REVIEWS,
        date_range=date_range,
    )

    if total == 0:
        return metrics

    metrics.by_relevance_bucket = _bucket_items(items, RELEVANCE_BUCKETS, "relevance_score")
    metrics.by_embedding_bucket = _bucket_items(items, EMBEDDING_BUCKETS, "embedding_score")

    # Per-source breakdown
    sources: dict[str, dict] = {}
    for item in items:
        name = item.source_name or "Unknown"
        if name not in sources:
            sources[name] = {"total": 0, "approved": 0}
        sources[name]["total"] += 1
        if item.approved:
            sources[name]["approved"] += 1
    for name, data in sources.items():
        data["rate"] = data["approved"] / data["total"] if data["total"] > 0 else 0.0
    metrics.by_source = sources

    # Optimal thresholds (only with sufficient data)
    metrics.optimal_relevance_threshold = _compute_optimal_threshold(
        items, "relevance_score", start=0.5, stop=1.0
    )
    metrics.optimal_embedding_threshold = _compute_optimal_threshold(
        items, "embedding_score", start=0.1, stop=0.8
    )

    metrics.weekly_trend = _compute_weekly_trend(items)

    return metrics


# ---------------------------------------------------------------------------
# Threshold suggestions
# ---------------------------------------------------------------------------

def suggest_thresholds(metrics: EvalMetrics, config: dict) -> list[str]:
    """Generate human-readable threshold adjustment suggestions."""
    suggestions: list[str] = []

    if not metrics.data_sufficient:
        suggestions.append(
            f"Only {metrics.total_reviewed} items reviewed — need at least "
            f"{MIN_REVIEWS} for reliable recommendations"
        )
        return suggestions

    current_settings = config.get("settings", {})

    # Relevance threshold suggestion
    current_rel = current_settings.get("relevance_threshold", 0.6)
    if metrics.optimal_relevance_threshold is not None:
        opt = metrics.optimal_relevance_threshold
        if abs(opt - current_rel) >= 0.05:
            # Calculate impact
            above = sum(
                d["total"] for d in metrics.by_relevance_bucket.values()
            )
            would_keep = sum(
                d["total"] for label, d in metrics.by_relevance_bucket.items()
                if float(label.split("-")[0].rstrip("+")) >= opt
            )
            reduction = 1 - (would_keep / above) if above > 0 else 0
            suggestions.append(
                f"Consider {'raising' if opt > current_rel else 'lowering'} "
                f"relevance threshold from {current_rel:.2f} to {opt:.2f} "
                f"(would change volume by {reduction:+.0%})"
            )

    # Embedding threshold suggestion
    current_emb = current_settings.get("similarity_threshold", 0.3)
    if metrics.optimal_embedding_threshold is not None:
        opt = metrics.optimal_embedding_threshold
        if abs(opt - current_emb) >= 0.05:
            suggestions.append(
                f"Consider {'raising' if opt > current_emb else 'lowering'} "
                f"embedding threshold from {current_emb:.2f} to {opt:.2f}"
            )

    # Per-source suggestions
    for name, data in metrics.by_source.items():
        if data["total"] >= MIN_BUCKET_SIZE and data["rate"] < 0.2:
            suggestions.append(
                f"{name} has a {data['rate']:.0%} approval rate "
                f"({data['total']} items) — consider raising its "
                f"similarity_threshold or reviewing its filter_keywords"
            )

    # Trend observation
    if len(metrics.weekly_trend) >= 3:
        recent = metrics.weekly_trend[-3:]
        rates = [w["rate"] for w in recent]
        if rates[-1] > rates[0] + 0.1:
            suggestions.append(
                "Approval rate is trending upward — embedding pre-filter "
                "is improving as SCMS grows"
            )
        elif rates[-1] < rates[0] - 0.1:
            suggestions.append(
                "Approval rate is trending downward — consider reviewing "
                "source configuration or threshold settings"
            )

    return suggestions


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _table_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def build_eval_report(metrics: EvalMetrics, suggestions: list[str]) -> str:
    """Assemble a markdown evaluation report."""
    today = date.today().isoformat()
    lines = [
        f"# Digest Evaluation Report — {today}",
        "",
        f"Generated: {today}",
        f"Data range: {metrics.date_range}",
        "",
    ]

    if not metrics.data_sufficient:
        lines.append(
            f"> **Note:** Only {metrics.total_reviewed} items reviewed. "
            f"Metrics below may not be statistically reliable "
            f"(minimum {MIN_REVIEWS} recommended)."
        )
        lines.append("")

    # Summary table
    lines.extend([
        "## Summary",
        "",
        _table_row(["Metric", "Value"]),
        _table_row(["---", "---"]),
        _table_row(["Total reviewed", str(metrics.total_reviewed)]),
        _table_row(["Approved", str(metrics.total_approved)]),
        _table_row(["Rejected", str(metrics.total_rejected)]),
        _table_row(["Overall approval rate", f"{metrics.approval_rate:.1%}"]),
        "",
    ])

    # Relevance buckets
    if metrics.by_relevance_bucket:
        lines.extend([
            "## Approval by Relevance Score",
            "",
            _table_row(["Bucket", "Total", "Approved", "Rate"]),
            _table_row(["---", "---", "---", "---"]),
        ])
        for label, data in metrics.by_relevance_bucket.items():
            lines.append(_table_row([
                label, str(data["total"]), str(data["approved"]),
                f"{data['rate']:.1%}" if data["total"] >= MIN_BUCKET_SIZE else
                f"{data['rate']:.1%} *",
            ]))
        lines.append("")
        lines.append("*\\* fewer than 5 items — interpret with caution*")
        lines.append("")

    # Embedding buckets
    if metrics.by_embedding_bucket:
        lines.extend([
            "## Approval by Embedding Score",
            "",
            _table_row(["Bucket", "Total", "Approved", "Rate"]),
            _table_row(["---", "---", "---", "---"]),
        ])
        for label, data in metrics.by_embedding_bucket.items():
            lines.append(_table_row([
                label, str(data["total"]), str(data["approved"]),
                f"{data['rate']:.1%}" if data["total"] >= MIN_BUCKET_SIZE else
                f"{data['rate']:.1%} *",
            ]))
        lines.append("")
        lines.append("*\\* fewer than 5 items — interpret with caution*")
        lines.append("")

    # Per-source breakdown
    if metrics.by_source:
        lines.extend([
            "## Per-Source Breakdown",
            "",
            _table_row(["Source", "Total", "Approved", "Rate"]),
            _table_row(["---", "---", "---", "---"]),
        ])
        for name, data in sorted(metrics.by_source.items(),
                                 key=lambda x: x[1]["total"], reverse=True):
            lines.append(_table_row([
                name, str(data["total"]), str(data["approved"]),
                f"{data['rate']:.1%}",
            ]))
        lines.append("")

    # Threshold analysis
    if metrics.optimal_relevance_threshold is not None or metrics.optimal_embedding_threshold is not None:
        lines.extend(["## Threshold Analysis", ""])
        if metrics.optimal_relevance_threshold is not None:
            lines.append(
                f"Optimal relevance threshold: **{metrics.optimal_relevance_threshold:.2f}**"
            )
        if metrics.optimal_embedding_threshold is not None:
            lines.append(
                f"Optimal embedding threshold: **{metrics.optimal_embedding_threshold:.2f}**"
            )
        lines.append("")

    # Weekly trend
    if metrics.weekly_trend:
        lines.extend([
            "## Weekly Trend",
            "",
            _table_row(["Week", "Total", "Approved", "Rate"]),
            _table_row(["---", "---", "---", "---"]),
        ])
        for week in metrics.weekly_trend:
            lines.append(_table_row([
                week["week"], str(week["total"]), str(week["approved"]),
                f"{week['rate']:.1%}",
            ]))
        lines.append("")

    # Recommendations
    if suggestions:
        lines.extend(["## Recommendations", ""])
        for s in suggestions:
            lines.append(f"- {s}")
        lines.append("")

    lines.extend([
        "---",
        f"*{metrics.total_reviewed} items from {metrics.date_range}.*",
    ])

    return "\n".join(lines)


def save_eval_report(markdown: str) -> Path:
    """Save the evaluation report to the digests directory."""
    digest_dir = Path(settings.digest_notes_dir).expanduser()
    digest_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{date.today().isoformat()}_eval_report.md"
    filepath = digest_dir / filename
    filepath.write_text(markdown)

    logger.info("Saved eval report to %s", filepath)
    return filepath


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def run_evaluation() -> dict:
    """Run the full digest evaluation pipeline.

    Returns:
        {"report_path": str, "total_reviewed": int, "approval_rate": float,
         "suggestions": list[str]}
    """
    logger.info("=== Starting digest evaluation ===")

    items = load_reviewed_items()

    if not items:
        logger.info("No reviewed items found")
        return {
            "report_path": "",
            "total_reviewed": 0,
            "approval_rate": 0.0,
            "suggestions": [],
        }

    metrics = compute_metrics(items)

    # Load current config for threshold comparison
    config_path = Path(settings.digest_config_path)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    suggestions = suggest_thresholds(metrics, config)
    report = build_eval_report(metrics, suggestions)
    report_path = save_eval_report(report)

    logger.info(
        "=== Evaluation complete: %d items, %.1f%% approval rate ===",
        metrics.total_reviewed,
        metrics.approval_rate * 100,
    )

    return {
        "report_path": str(report_path),
        "total_reviewed": metrics.total_reviewed,
        "approval_rate": metrics.approval_rate,
        "suggestions": suggestions,
    }
