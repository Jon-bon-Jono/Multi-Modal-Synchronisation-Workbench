@echo off
setlocal
pushd "%~dp0..\.." || exit /b 1

python ml\mmfi_pose_quick.py finetune-syncwb ^
  --checkpoint "runs\mmfi_pose_anchor_v4_rank_signal\best.pt" ^
  --sqlite "workbench.sqlite" ^
  --artifact-root "artifact_store" ^
  --subjects "07_SW" ^
  --mapping-methods "initial_nearest_for_anchoring" ^
  --out "runs\mmfi_pose_anchor_v4_rank_signal_finetuned_w9_07_SW" ^
  --epochs 10 ^
  --batch-size 64 ^
  --workers 0 ^
  --lr 3e-5 ^
  --freeze-backbone-epochs 1

set "SYNCWB_EXIT_CODE=%ERRORLEVEL%"
popd
endlocal & exit /b %SYNCWB_EXIT_CODE%
