@echo off
setlocal
pushd "%~dp0..\.." || exit /b 1

syncwb fit-piecewise ^
  --sqlite workbench.sqlite ^
  --subject 19_MM ^
  --source-run "Session-2024-January-15 09-47-41-274452" ^
  --source-device kinect_rgb ^
  --source-timeline rgb_wallclock_from_pts ^
  --target-run "Session-2024-January-15 09-43-27-126274" ^
  --target-device radar_pc ^
  --target-timeline radar_pc_linear_from_index ^
  --sync-model piecewise_rgb_to_pc_v001 ^
  --mapping-version piecewise_rgb_to_pc_v001_map ^
  --parent-mapping-version "initial_rgb_to_pc_v001__19_MM__Session-2024-January-15_09-47-41-274452__Session-2024-January-15_09-43-27-126274" ^
  --top-k 3 ^
  --extrapolation-policy disallow ^
  --primary-policy supported-only ^
  --diagnostics-csv reports/piecewise_rgb_to_pc_v001.csv

set "SYNCWB_EXIT_CODE=%ERRORLEVEL%"
popd

REM sync-model is name (id) of new sync-model
REM mapping-version is name (id) of new mapping version
REM parent-mapping-version is original mapping version enterred into the GUI
REM first 07_SW: initial_rgb_to_pc_v001__07_SW__Session-2023-November-27_14-11-02-690792__Session-2023-November-27_13-59-35-723243
REM 19_MM: initial_rgb_to_pc_v001__19_MM__Session-2024-January-15_09-47-41-274452__Session-2024-January-15_09-43-27-126274

endlocal & exit /b %SYNCWB_EXIT_CODE%
