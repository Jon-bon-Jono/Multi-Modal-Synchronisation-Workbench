"""Read-only launch choices and immutable cloud selection for one GUI session."""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3


@dataclass(frozen=True)
class CloudSessionSelection:
    subject_id: str
    mapping_version_id: str
    source_run_id: str
    target_run_id: str
    target_device_type: str
    point_cloud_version_id: str
    readable_label: str
    artifact_ref: str
    payload_fingerprint: str = ""
    acquisition_timeline_sha256: str = ""

    @property
    def source_label(self):
        name = "Offline raw" if self.target_device_type == "radar_raw" else "Online"
        return f"{name} — {self.target_run_id}"

    def cloud_provenance(self):
        return {"subject_id": self.subject_id, "run_id": self.target_run_id,
                "device_type": self.target_device_type, "point_cloud_version_id": self.point_cloud_version_id,
                "readable_label": self.readable_label, "payload_fingerprint": self.payload_fingerprint,
                "acquisition_timeline_sha256": self.acquisition_timeline_sha256}


def session_choices(sqlite_path, artifact_root, subject_id) -> list[CloudSessionSelection]:
    """Only RGB-to-radar mappings with a usable, supported cloud bundle are offered.

    Legacy online databases remain readable without an implicit migration.
    Raw legacy payloads have no tracked provenance and are not GUI candidates.
    """
    with closing(sqlite3.connect(Path(sqlite_path).resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        mappings = conn.execute("SELECT * FROM MAPPING_VERSION WHERE subject_id=? AND source_device_type='kinect_rgb' AND target_device_type IN ('radar_pc','radar_raw')", (subject_id,)).fetchall()
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "POINT_CLOUD_VERSION" in tables:
            versions = [dict(r) for r in conn.execute("SELECT * FROM POINT_CLOUD_VERSION WHERE subject_id=?", (subject_id,))]
        else:
            versions = [dict(r, point_cloud_version_id="online_original", readable_label="Original online cloud")
                        for r in conn.execute("SELECT DISTINCT subject_id,run_id,device_type,artifact_ref FROM SAMPLE_ARTIFACT WHERE subject_id=? AND device_type='radar_pc' AND artifact_role='radar_points'", (subject_id,))]
        choices = []
        for mapping in mappings:
            if conn.execute("SELECT 1 FROM SAMPLE_MAPPING WHERE subject_id=? AND mapping_version_id=? LIMIT 1", (subject_id, mapping["mapping_version_id"])).fetchone() is None:
                continue
            for v in versions:
                if (v["run_id"], v["device_type"]) != (mapping["target_run_id"], mapping["target_device_type"]):
                    continue
                if not v.get("artifact_ref") or not (Path(artifact_root) / v["artifact_ref"]).is_file():
                    continue
                if v["device_type"] == "radar_pc":
                    if v["point_cloud_version_id"] != "online_original":
                        continue
                else:
                    try:
                        manifest = json.loads(v["provenance_json"])["version"]
                        if manifest["tracking_enabled"] is not True or manifest["point_cloud_version_id"] != v["point_cloud_version_id"]:
                            continue
                    except (KeyError, TypeError, ValueError):
                        continue
                choices.append(CloudSessionSelection(subject_id=subject_id, mapping_version_id=mapping["mapping_version_id"],
                    source_run_id=mapping["source_run_id"], target_run_id=v["run_id"], target_device_type=v["device_type"],
                    point_cloud_version_id=v["point_cloud_version_id"], readable_label=v["readable_label"], artifact_ref=v["artifact_ref"],
                    payload_fingerprint=v.get("payload_fingerprint") or "", acquisition_timeline_sha256=v.get("acquisition_timeline_sha256") or ""))
    return sorted(choices, key=lambda c: (c.target_device_type, c.target_run_id, c.readable_label, c.mapping_version_id))


def resolve_session(sqlite_path, artifact_root, subject_id, mapping_version_id, point_cloud_version_id=None):
    matches = [c for c in session_choices(sqlite_path, artifact_root, subject_id) if c.mapping_version_id == mapping_version_id]
    if point_cloud_version_id is None:
        # Compatibility for existing online-only Python callers. GUI selection is explicit.
        matches = [c for c in matches if c.target_device_type == "radar_pc" and c.point_cloud_version_id == "online_original"]
    else:
        matches = [c for c in matches if c.point_cloud_version_id == point_cloud_version_id]
    if len(matches) != 1:
        raise ValueError("Choose a supported cloud version and a compatible RGB-to-radar mapping with available artifacts. Raw versions must be selected explicitly.")
    return matches[0]


def make_session_dialog_class():
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QVBoxLayout

    class SessionDialog(QDialog):
        def __init__(self, choices, *, preferred_mapping=None, preferred_version=None, parent=None):
            super().__init__(parent)
            self.choices = choices
            self.preferred_mapping = preferred_mapping
            self.preferred_version = preferred_version
            self.setWindowTitle("SyncWB — start anchoring session")
            self.resize(860, 330)
            layout = QVBoxLayout(self)
            intro = QLabel("Choose the point cloud to view. This selection stays fixed until you close the anchoring session.")
            intro.setWordWrap(True)
            layout.addWidget(intro)
            form = QFormLayout()
            self.source_combo, self.version_combo, self.mapping_combo = QComboBox(), QComboBox(), QComboBox()
            for combo in (self.source_combo, self.version_combo, self.mapping_combo):
                combo.setMinimumContentsLength(30)
                combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            form.addRow("Point-cloud source", self.source_combo)
            form.addRow("Cloud version", self.version_combo)
            form.addRow("Synchronization mapping", self.mapping_combo)
            layout.addLayout(form)
            self.details = QLabel()
            self.details.setWordWrap(True)
            self.details.setTextInteractionFlags(Qt.TextSelectableByMouse)
            layout.addWidget(self.details)
            self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            self.buttons.button(QDialogButtonBox.Ok).setText("Start session")
            self.buttons.accepted.connect(self.accept)
            self.buttons.rejected.connect(self.reject)
            layout.addWidget(self.buttons)
            seen = set()
            for choice in choices:
                key = json.dumps([choice.target_device_type, choice.target_run_id])
                if key not in seen:
                    self.source_combo.addItem(choice.source_label, key)
                    seen.add(key)
            preferred = next((c for c in choices if (not preferred_mapping or c.mapping_version_id == preferred_mapping)
                              and (not preferred_version or c.point_cloud_version_id == preferred_version)), None)
            if preferred:
                self.source_combo.setCurrentIndex(self.source_combo.findData(json.dumps([preferred.target_device_type, preferred.target_run_id])))
            self.source_combo.currentIndexChanged.connect(self._source_changed)
            self.version_combo.currentIndexChanged.connect(self._version_changed)
            self.mapping_combo.currentIndexChanged.connect(self._update_details)
            self._source_changed()

        def _source_choices(self):
            return [c for c in self.choices if json.dumps([c.target_device_type, c.target_run_id]) == self.source_combo.currentData()]

        def _source_changed(self, *_):
            self.version_combo.blockSignals(True)
            self.version_combo.clear()
            seen = set()
            for choice in self._source_choices():
                if choice.point_cloud_version_id not in seen:
                    self.version_combo.addItem(choice.readable_label + " — " + choice.point_cloud_version_id[:16], choice.point_cloud_version_id)
                    seen.add(choice.point_cloud_version_id)
            preferred = self.version_combo.findData(self.preferred_version)
            if preferred >= 0:
                self.version_combo.setCurrentIndex(preferred)
            self.version_combo.blockSignals(False)
            self._version_changed()

        def _version_changed(self, *_):
            self.mapping_combo.blockSignals(True)
            self.mapping_combo.clear()
            for choice in self._source_choices():
                if choice.point_cloud_version_id == self.version_combo.currentData():
                    self.mapping_combo.addItem(choice.mapping_version_id, choice)
            preferred = next((i for i in range(self.mapping_combo.count()) if self.mapping_combo.itemData(i).mapping_version_id == self.preferred_mapping), -1)
            if preferred >= 0:
                self.mapping_combo.setCurrentIndex(preferred)
            self.mapping_combo.blockSignals(False)
            self._update_details()

        def _update_details(self, *_):
            choice = self.selection
            self.buttons.button(QDialogButtonBox.Ok).setEnabled(choice is not None)
            if choice:
                self.details.setText(f"RGB: {choice.source_run_id}\nCloud ID: {choice.point_cloud_version_id}\nAnchors remain compatible with other versions of this same acquisition.")
            else:
                self.details.setText("No compatible mapped cloud is available. Import a supported cloud and create an RGB-to-radar navigation mapping before starting.")

        @property
        def selection(self):
            return self.mapping_combo.currentData()

    return SessionDialog
