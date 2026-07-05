"""Service for writing and exporting canonical anchors."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from sync_workbench.core.ids import slugify
from sync_workbench.core.tables import align_to_spec
from sync_workbench.core.time_utils import utc_now_str
from sync_workbench.storage.sqlite_store import SQLiteCoreStore


@dataclass(frozen=True)
class AnchorEndpoint:
    run_id: str
    device_type: str
    sample_index: int
    member_role: str
    confidence: float | None = None
    notes: str = ""


class AnchorService:
    def __init__(self, sqlite_path: str | Path):
        self.store = SQLiteCoreStore(sqlite_path)

    def create_pair_anchor(
        self,
        *,
        subject_id: str,
        source: AnchorEndpoint,
        target: AnchorEndpoint,
        anchor_id: str | None = None,
        anchor_type: str = "manual_correspondence",
        label: str = "",
        confidence: float | None = None,
        user_notes: str = "",
        provenance: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> str:
        anchor_id = anchor_id or self._new_anchor_id(subject_id)
        if self.store.row_exists("ANCHOR", {"subject_id": subject_id, "anchor_id": anchor_id}) and not overwrite:
            raise ValueError(f"Anchor already exists: {subject_id}/{anchor_id}")

        notes_payload = {
            "user_notes": user_notes or "",
            "provenance": {
                "created_by": "anchor_service",
                "created_at": utc_now_str(),
                **(provenance or {}),
            },
        }
        anchor_df = align_to_spec(
            "ANCHOR",
            pd.DataFrame(
                [
                    {
                        "subject_id": subject_id,
                        "anchor_id": anchor_id,
                        "anchor_type": anchor_type,
                        "label": label,
                        "confidence": "" if confidence is None else float(confidence),
                        "notes": json.dumps(notes_payload, sort_keys=True),
                    }
                ]
            ),
        )
        member_rows = []
        for endpoint in (source, target):
            member_rows.append(
                {
                    "subject_id": subject_id,
                    "anchor_id": anchor_id,
                    "run_id": endpoint.run_id,
                    "device_type": endpoint.device_type,
                    "sample_index": int(endpoint.sample_index),
                    "member_role": endpoint.member_role,
                    "confidence": "" if endpoint.confidence is None else float(endpoint.confidence),
                    "notes": endpoint.notes,
                }
            )
        member_df = align_to_spec("ANCHOR_MEMBER", pd.DataFrame(member_rows))

        if overwrite:
            self.delete_anchor(subject_id, anchor_id, missing_ok=True)
        self.store.replace_or_append_without_key_conflicts("ANCHOR", anchor_df)
        self.store.replace_or_append_without_key_conflicts("ANCHOR_MEMBER", member_df)
        return anchor_id

    def delete_anchor(self, subject_id: str, anchor_id: str, *, missing_ok: bool = False) -> None:
        found = self.store.row_exists("ANCHOR", {"subject_id": subject_id, "anchor_id": anchor_id})
        if not found and not missing_ok:
            raise KeyError(f"Anchor not found: {subject_id}/{anchor_id}")
        self.store.delete_where("ANCHOR_MEMBER", {"subject_id": subject_id, "anchor_id": anchor_id})
        self.store.delete_where("MODEL_ANCHOR", {"subject_id": subject_id, "anchor_id": anchor_id})
        self.store.delete_where("ANCHOR", {"subject_id": subject_id, "anchor_id": anchor_id})

    def list_pair_anchors(
        self,
        *,
        subject_id: str,
        source_run_id: str,
        source_device_type: str,
        target_run_id: str,
        target_device_type: str,
    ) -> pd.DataFrame:
        anchors = self.store.read_table("ANCHOR")
        members = self.store.read_table("ANCHOR_MEMBER")
        if anchors.empty or members.empty:
            return pd.DataFrame()
        members = members[members["subject_id"].astype(str) == str(subject_id)].copy()
        src = members[
            (members["run_id"].astype(str) == str(source_run_id))
            & (members["device_type"].astype(str) == str(source_device_type))
        ].copy()
        tgt = members[
            (members["run_id"].astype(str) == str(target_run_id))
            & (members["device_type"].astype(str) == str(target_device_type))
        ].copy()
        if src.empty or tgt.empty:
            return pd.DataFrame()
        paired = src.merge(tgt, on=["subject_id", "anchor_id"], suffixes=("_source", "_target"))
        anchors_small = anchors[anchors["subject_id"].astype(str) == str(subject_id)].copy()
        return paired.merge(anchors_small, on=["subject_id", "anchor_id"], how="left")

    def export_pair_anchors_json(
        self,
        path: str | Path,
        *,
        subject_id: str,
        source_run_id: str,
        source_device_type: str,
        target_run_id: str,
        target_device_type: str,
        session_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        anchors = self.store.read_table("ANCHOR")
        members = self.store.read_table("ANCHOR_MEMBER")
        pair = self.list_pair_anchors(
            subject_id=subject_id,
            source_run_id=source_run_id,
            source_device_type=source_device_type,
            target_run_id=target_run_id,
            target_device_type=target_device_type,
        )
        anchor_ids = sorted(set(pair["anchor_id"].astype(str))) if not pair.empty else []
        payload = {
            "exported_at": utc_now_str(),
            "session": session_metadata or {},
            "pair": {
                "subject_id": subject_id,
                "source_run_id": source_run_id,
                "source_device_type": source_device_type,
                "target_run_id": target_run_id,
                "target_device_type": target_device_type,
            },
            "ANCHOR": anchors[
                (anchors["subject_id"].astype(str) == str(subject_id))
                & (anchors["anchor_id"].astype(str).isin(anchor_ids))
            ].to_dict("records"),
            "ANCHOR_MEMBER": members[
                (members["subject_id"].astype(str) == str(subject_id))
                & (members["anchor_id"].astype(str).isin(anchor_ids))
            ].to_dict("records"),
        }
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return payload

    def import_anchors_json(self, path: str | Path, *, overwrite: bool = False) -> dict[str, int]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        anchors = align_to_spec("ANCHOR", pd.DataFrame(payload.get("ANCHOR", [])))
        members = align_to_spec("ANCHOR_MEMBER", pd.DataFrame(payload.get("ANCHOR_MEMBER", [])))
        if overwrite:
            for row in anchors.to_dict("records"):
                self.delete_anchor(str(row["subject_id"]), str(row["anchor_id"]), missing_ok=True)
        self.store.replace_or_append_without_key_conflicts("ANCHOR", anchors)
        self.store.replace_or_append_without_key_conflicts("ANCHOR_MEMBER", members)
        return {"anchors": int(len(anchors)), "members": int(len(members))}

    @staticmethod
    def _new_anchor_id(subject_id: str) -> str:
        return f"anchor_{slugify(subject_id, max_len=30)}_{utc_now_str().replace(':', '').replace('.', '')}_{uuid.uuid4().hex[:8]}"
