@echo off
setlocal
pushd "%~dp0..\.." || exit /b 1

python ml\mmfi_pose_quick.py infer-syncwb-all ^
  --checkpoint "runs\mmfi_pose_anchor_v4_rank_signal_finetuned_w9_07_SW\best.pt" ^
  --sqlite "workbench.sqlite" ^
  --artifact-root "artifact_store" ^
  --subjects "19_MM" ^
  --mapping-methods "nearest_predicted_time" ^
  --out "runs\19_MM_mmfi_pose_anchor_v4_rank_signal_finetuned_w9_07_SW"

set "SYNCWB_EXIT_CODE=%ERRORLEVEL%"
popd
endlocal & exit /b %SYNCWB_EXIT_CODE%
