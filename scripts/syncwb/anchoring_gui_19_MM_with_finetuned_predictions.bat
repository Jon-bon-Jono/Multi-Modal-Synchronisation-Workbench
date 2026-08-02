@echo off
setlocal
pushd "%~dp0..\.." || exit /b 1

syncwb anchoring-gui ^
  --sqlite workbench.sqlite ^
  --artifact-root artifact_store ^
  --rgb-root "E:/smart_cup_recordings/Kinect" ^
  --subject 19_MM ^
  --mapping-version piecewise_rgb_to_pc_v001_map ^
  --pose-predictions "runs/19_MM_mmfi_pose_anchor_v4_robust_signal_finetuned_07_SW/19_MM/piecewise_rgb_to_pc_v001_map/predictions.npz" ^
  --pose-prediction-array pred ^
  --annotator-id JW01

set "SYNCWB_EXIT_CODE=%ERRORLEVEL%"
popd
endlocal & exit /b %SYNCWB_EXIT_CODE%
