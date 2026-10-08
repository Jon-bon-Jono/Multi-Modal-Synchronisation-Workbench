@echo off
setlocal
rem Online-cloud preset: desk calibration applies only to offline raw radar.
set "SYNCWB_SPATIAL_CALIBRATION="
call "%~dp0run_source_gui.cmd" ^
  --subject 19_MM ^
  --mapping-version piecewise_rgb_to_pc_v001_map ^
  --annotator-id JW01 %*
exit /b %ERRORLEVEL%
