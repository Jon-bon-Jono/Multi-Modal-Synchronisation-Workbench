"""Service for writing and exporting canonical anchors."""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from sync_workbench.services.anchor_transfer import (
    SCHEMA, acquisition_digest, atomic_json, equivalent, insert_rows,
    normalized_rows, validate_payload,
)
from sync_workbench.deployment.package_layout import json_digest
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

        anchors = normalized_rows("ANCHOR", anchor_df.to_dict("records"))
        members = normalized_rows("ANCHOR_MEMBER", member_df.to_dict("records"))
        pair = {"subject_id":subject_id, "source_run_id":source.run_id, "source_device_type":source.device_type,
                "target_run_id":target.run_id, "target_device_type":target.device_type}
        with closing(self.store.connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            validate_payload({"pair":pair,"ANCHOR":anchors,"ANCHOR_MEMBER":members},conn)
            if conn.execute("SELECT 1 FROM ANCHOR WHERE subject_id=? AND anchor_id=?",(subject_id,anchor_id)).fetchone():
                if not overwrite:
                    raise ValueError(f"Anchor already exists: {subject_id}/{anchor_id}")
                self._delete(conn,subject_id,anchor_id)
            insert_rows(conn,"ANCHOR",anchors)
            insert_rows(conn,"ANCHOR_MEMBER",members)
        return anchor_id

    @staticmethod
    def _delete(conn, subject_id, anchor_id):
        if conn.execute("SELECT 1 FROM MODEL_ANCHOR WHERE subject_id=? AND anchor_id=?",(subject_id,anchor_id)).fetchone():
            raise ValueError("Anchor is used by a synchronization model; retain it and create a new anchor instead")
        for table in ("ANCHOR_MEMBER","ANCHOR"):
            conn.execute(f'DELETE FROM "{table}" WHERE subject_id=? AND anchor_id=?',(subject_id,anchor_id))

    def delete_anchor(self, subject_id: str, anchor_id: str, *, missing_ok: bool = False) -> None:
        with closing(self.store.connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            found = conn.execute("SELECT 1 FROM ANCHOR WHERE subject_id=? AND anchor_id=?",(subject_id,anchor_id)).fetchone()
            if not found and not missing_ok:
                raise KeyError(f"Anchor not found: {subject_id}/{anchor_id}")
            self._delete(conn,subject_id,anchor_id)

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
        out_path = Path(path).resolve()
        if out_path == self.store.path.resolve():
            raise ValueError("An export cannot overwrite its working database")
        pair = {"subject_id":subject_id,"source_run_id":source_run_id,"source_device_type":source_device_type,
                "target_run_id":target_run_id,"target_device_type":target_device_type}
        with closing(sqlite3.connect(self.store.path.resolve().as_uri()+"?mode=ro",uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN")
            anchors = [dict(r) for r in conn.execute("SELECT * FROM ANCHOR WHERE subject_id=? ORDER BY anchor_id",(subject_id,))]
            members = [dict(r) for r in conn.execute("SELECT * FROM ANCHOR_MEMBER WHERE subject_id=? ORDER BY anchor_id,member_role",(subject_id,))]
            source_ids = {r['anchor_id'] for r in members if (r['run_id'],r['device_type'])==(source_run_id,source_device_type)}
            target_ids = {r['anchor_id'] for r in members if (r['run_id'],r['device_type'])==(target_run_id,target_device_type)}
            ids = source_ids & target_ids
            payload = {"schema":SCHEMA,"exported_at":utc_now_str(),"session":session_metadata or {},"pair":pair,
                       "acquisition_sha256":acquisition_digest(conn,pair),
                       "ANCHOR":normalized_rows("ANCHOR",[r for r in anchors if r['anchor_id'] in ids]),
                       "ANCHOR_MEMBER":normalized_rows("ANCHOR_MEMBER",[r for r in members if r['anchor_id'] in ids])}
        payload["content_sha256"] = json_digest(payload)
        atomic_json(out_path,payload)
        return payload

    def import_anchors_json(self, path: str | Path, *, overwrite: bool = False,
                            expected_manifest: str | Path | None = None, dry_run: bool = False) -> dict[str, int]:
        if overwrite:
            raise ValueError("Anchor imports never overwrite conflicts; resolve the conflicting ID explicitly")
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        # All validation and conflict checks share one transaction with the inserts.
        with closing(sqlite3.connect(self.store.path.resolve().as_uri()+"?mode=rw",uri=True)) as conn, conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            anchors, members = validate_payload(payload,conn,expected_manifest)
            pending, unchanged, conflicts = [], 0, []
            for anchor in anchors:
                key = (anchor['subject_id'],anchor['anchor_id'])
                old = normalized_rows("ANCHOR",[dict(r) for r in conn.execute("SELECT * FROM ANCHOR WHERE subject_id=? AND anchor_id=?",key)])
                old_members = normalized_rows("ANCHOR_MEMBER",[dict(r) for r in conn.execute("SELECT * FROM ANCHOR_MEMBER WHERE subject_id=? AND anchor_id=?",key)])
                if old or old_members:
                    if equivalent(old,[anchor]) and equivalent(old_members,members[key]):
                        unchanged += 1
                    else:
                        conflicts.append('/'.join(key))
                else:
                    pending.append(anchor)
            if conflicts:
                raise ValueError("Anchor ID conflicts; nothing imported: " + ", ".join(conflicts))
            pending_members = [m for a in pending for m in members[(a['subject_id'],a['anchor_id'])]]
            if not dry_run:
                insert_rows(conn,"ANCHOR",pending)
                insert_rows(conn,"ANCHOR_MEMBER",pending_members)
            return {"anchors":len(pending),"members":len(pending_members),"unchanged":unchanged,"dry_run":dry_run}

    @staticmethod
    def _new_anchor_id(subject_id: str) -> str:
        return f"anchor_{slugify(subject_id, max_len=30)}_{utc_now_str().replace(':', '').replace('.', '')}_{uuid.uuid4().hex[:8]}"
