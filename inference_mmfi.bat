python mmfi_pose_quick.py infer-syncwb ^
  --checkpoint "runs\mmfi_pose_anchor_v4\best.pt" ^
  --sqlite "workbench.sqlite" ^
  --artifact-root "artifact_store" ^
  --subject "19_MM" ^
  --mapping-version "piecewise_rgb_to_pc_v001_map" ^
  --out "runs\19_MM_mmfi_pose_anchor_v4"
