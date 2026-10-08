@echo off
setlocal DisableDelayedExpansion
rem Edit these presets here; calibration is not a launch-time argument.
set "SYNCWB_SPATIAL_CALIBRATION=%~dp0..\..\calibration\kinect_radar\2026-10-06-desk\desk_all.json"
set "SYNCWB_EXPORT_OUTPUT=%USERPROFILE%\Documents\SyncWB\training_exports\19_MM_dense_static_far_initial_v001_desk_all"
pushd "%~dp0..\.." || exit /b 1
set "PYTHONPATH=%CD%\src"
set "PYTHONNOUSERSITE=1"
set "PYTHONDONTWRITEBYTECODE=1"
if not defined SYNCWB_CONDA_EXE set "SYNCWB_CONDA_EXE=%USERPROFILE%\anaconda3\Scripts\conda.exe"
if not defined SYNCWB_SOURCE_ENV set "SYNCWB_SOURCE_ENV=%USERPROFILE%\anaconda3\envs\syncwb"
if not exist "%SYNCWB_CONDA_EXE%" (
    echo Cannot find Conda. Set SYNCWB_CONDA_EXE to its executable path and retry.
    goto failed
)
if not exist "%SYNCWB_SPATIAL_CALIBRATION%" (
    echo Cannot find the preset spatial calibration: "%SYNCWB_SPATIAL_CALIBRATION%"
    goto failed
)
call "%SYNCWB_CONDA_EXE%" run --no-capture-output --prefix "%SYNCWB_SOURCE_ENV%" python -m sync_workbench.cli.main export-training-data ^
  --sqlite "%CD%\workbench.sqlite" ^
  --artifact-root "%CD%\artifact_store" ^
  --subject 19_MM ^
  --mapping-version initial_rgb_to_raw_v001 ^
  --point-cloud-version raw_d3f274e06a775b757e89dc86f8f57dfb12ab4a198e66912af21dd176d39620f3 ^
  --read-only-root "D:\smart_cup_recordings" ^
  --output "%SYNCWB_EXPORT_OUTPUT%" %* ^
  --spatial-calibration "%SYNCWB_SPATIAL_CALIBRATION%"
set "SYNCWB_EXIT_CODE=%ERRORLEVEL%"
if not "%SYNCWB_EXIT_CODE%"=="0" if not defined SYNCWB_NO_PAUSE pause
popd
endlocal & exit /b %SYNCWB_EXIT_CODE%
:failed
if not defined SYNCWB_NO_PAUSE pause
popd
exit /b 1
