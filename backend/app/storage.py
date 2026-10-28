"""SQLite storage layer for persistent experiments and traces."""
import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from app.config import settings


def _get_db_path() -> str:
    url = settings.DATABASE_URL
    if url.startswith("sqlite:///"):
        path = url.replace("sqlite:///", "")
        if path.startswith("./"):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            return os.path.join(base_dir, path[2:])
        return path
    return "experiments.db"


class Storage:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or _get_db_path()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    experiment_type TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    model TEXT NOT NULL,
                    layer INTEGER,
                    head INTEGER,
                    target_token TEXT,
                    metrics_summary TEXT,
                    data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS traces (
                    id TEXT PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    model TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_exp_created ON experiments (created_at DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_exp_type ON experiments (experiment_type)")
            conn.commit()

    def save_experiment(self, exp: Dict[str, Any]) -> Dict[str, Any]:
        exp_id = str(exp.get("experiment_id") or exp.get("id"))
        exp_type = exp.get("intervention", {}).get("type") or exp.get("experiment_type", "head_ablation")
        prompt = exp.get("prompt", "")
        model = exp.get("model", settings.MODEL_NAME)
        layer = exp.get("layer")
        head = exp.get("head")
        target_token = (
            exp.get("baseline", {}).get("target_token")
            or exp.get("target_token")
            or ""
        )
        
        # Build human-readable metrics summary
        metrics = exp.get("metrics", {})
        summary_parts = []
        if "target_probability_delta" in metrics:
            summary_parts.append(f"Δprob: {metrics['target_probability_delta'] * 100:+.2f}%")
        if "target_logit_delta" in metrics:
            summary_parts.append(f"Δlogit: {metrics['target_logit_delta']:+.3f}")
        if "rank_delta" in metrics:
            summary_parts.append(f"Δrank: {metrics['rank_delta']:+d}")
        if "causal_recovery" in metrics:
            summary_parts.append(f"Recovery: {metrics['causal_recovery'] * 100:.1f}%")
        if "top_causal_head" in metrics:
            summary_parts.append(f"Top Head: {metrics['top_causal_head']}")
        metrics_summary = " | ".join(summary_parts) if summary_parts else "Done"

        created_at = exp.get("created_at") or datetime.now(timezone.utc).isoformat()
        exp["created_at"] = created_at

        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO experiments (
                    id, experiment_type, prompt, model, layer, head, target_token, metrics_summary, data_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                exp_id, exp_type, prompt, model, layer, head, target_token,
                metrics_summary, json.dumps(exp), created_at
            ))
            conn.commit()
        return exp

    def get_experiment(self, exp_id: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            row = conn.execute("SELECT data_json FROM experiments WHERE id = ?", (exp_id,)).fetchone()
            if row:
                return json.loads(row["data_json"])
        return None

    def list_experiments(self, limit: int = 50, exp_type: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            if exp_type:
                rows = conn.execute("""
                    SELECT id, experiment_type, prompt, model, layer, head, target_token, metrics_summary, created_at
                    FROM experiments
                    WHERE experiment_type = ?
                    ORDER BY created_at DESC LIMIT ?
                """, (exp_type, limit)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT id, experiment_type, prompt, model, layer, head, target_token, metrics_summary, created_at
                    FROM experiments
                    ORDER BY created_at DESC LIMIT ?
                """, (limit,)).fetchall()
            return [dict(r) for r in rows]

    def delete_experiment(self, exp_id: str) -> bool:
        with self._get_conn() as conn:
            cur = conn.execute("DELETE FROM experiments WHERE id = ?", (exp_id,))
            conn.commit()
            return cur.rowcount > 0

    def save_trace(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        trace_id = trace["trace_id"]
        created_at = trace.get("created_at") or datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO traces (id, prompt, model, data_json, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                trace_id,
                trace.get("input_prompt", ""),
                settings.MODEL_NAME,
                json.dumps(trace),
                created_at
            ))
            conn.commit()
        return trace

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            row = conn.execute("SELECT data_json FROM traces WHERE id = ?", (trace_id,)).fetchone()
            if row:
                return json.loads(row["data_json"])
        return None


storage = Storage()
