@echo off
setlocal
pushd "%~dp0..\.." || exit /b 1

python ml\mmfi_pose_quick.py train ^
  --packed-root "C:\Users\GSBME\SmartCupStudy\Unified_network\data_sets\MM-Fi\packed_data" ^
  --out "runs\mmfi_pose_anchor_v4_rank_signal_w9" ^
  --split cross_environment ^
  --spatial-mode cloud_anchor ^
  --target-mode cloud_anchor_relative ^
  --signal-mode rank ^
  --num-points 450 ^
  --window-size 9 ^
  --sample-period-sec 0.1 ^
  --epochs 25 ^
  --batch-size 128 ^
  --workers 4

set "SYNCWB_EXIT_CODE=%ERRORLEVEL%"
popd
endlocal & exit /b %SYNCWB_EXIT_CODE%
