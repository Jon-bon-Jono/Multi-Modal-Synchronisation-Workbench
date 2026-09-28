@echo off
setlocal
pushd "%~dp0..\.." || exit /b 1

rem Uses the prepared validation database, never the original live database.
rem Kinect recordings are read only. Start dialog allows online or offline raw.
syncwb anchoring-gui ^
  --sqlite "%USERPROFILE%\Documents\SyncWB\backend_validation\workbench_raw_validation.sqlite" ^
  --artifact-root "%USERPROFILE%\Documents\SyncWB\backend_validation\artifact_store" ^
  --rgb-root "D:/smart_cup_recordings/Kinect" ^
  --subject 19_MM ^
  --mapping-version initial_rgb_to_raw_v001 ^
  --annotator-id JW01

set "SYNCWB_EXIT_CODE=%ERRORLEVEL%"
popd
endlocal & exit /b %SYNCWB_EXIT_CODE%
