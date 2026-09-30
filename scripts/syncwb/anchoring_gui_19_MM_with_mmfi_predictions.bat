@echo off
setlocal
call "%~dp0run_source_gui.cmd" ^
  --subject 19_MM ^
  --mapping-version piecewise_rgb_to_pc_v001_map ^
  --pose-predictions "runs/19_MM_mmfi_pose_anchor_v4/predictions.npz" ^
  --pose-prediction-array pred ^
  --annotator-id JW01 %*
exit /b %ERRORLEVEL%
