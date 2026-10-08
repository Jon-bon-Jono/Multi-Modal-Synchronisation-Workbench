@echo off
setlocal
rem Edit this preset to select another Kinect/raw-radar calibration.
set "SYNCWB_SPATIAL_CALIBRATION=%~dp0..\..\calibration\kinect_radar\2026-10-06-desk\desk_all.json"
call "%~dp0run_source_gui.cmd" ^
  --subject 09_SY ^
  --mapping-version initial_rgb_to_raw_v001 ^
  --point-cloud-version raw_fbf432d2908ce574ec3daa437ab78762c7ed850bac3aeb6e7cb8b1aef9aeae7c ^
  --annotator-id JW01 %*
exit /b %ERRORLEVEL%
