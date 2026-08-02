@echo off
setlocal
pushd "%~dp0..\.." || exit /b 1

syncwb map-nearest-all ^
  --sqlite "workbench.sqlite" ^
  --source-device "kinect_rgb" ^
  --source-timeline "rgb_wallclock_from_pts" ^
  --target-device "radar_pc" ^
  --target-timeline "radar_pc_linear_from_index" ^
  --mapping-version-prefix "initial_rgb_to_pc_v001" ^
  --top-k 3 ^
  --min-overlap-sec 5 ^
  --source-window-policy "target-overlap" ^
  --source-margin-ms 100 ^
  --primary-policy "supported-only" ^
  --pair-report-csv "reports/initial_rgb_to_pc_v001_pair_report.csv" ^
  --overwrite

set "SYNCWB_EXIT_CODE=%ERRORLEVEL%"
popd
endlocal & exit /b %SYNCWB_EXIT_CODE%
