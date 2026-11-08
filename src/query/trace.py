from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import Settings, settings as default_settings


class TraceWriter:
    def __init__(
        self,
        trace_dir: Optional[Path] = None,
        settings: Settings = default_settings,
    ) -> None:
        self.trace_dir = Path(trace_dir) if trace_dir is not None else Path(
            settings.trace_dir
        )

    def write(self, payload: Dict[str, Any]) -> Path:
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        trace_id = uuid.uuid4().hex
        path = self.trace_dir / f"{timestamp}-{trace_id}.json"
        data = dict(payload)
        data.setdefault("trace_id", trace_id)
        data.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return path
