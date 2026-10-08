@echo off
setlocal DisableDelayedExpansion
rem Run with --dry-run to preview coverage; run without it to create the export.
rem Override output via SYNCWB_EXPORT_OUTPUT or --output NEW_DIRECTORY.
rem Edit the selection JSON to choose subject-specific mapping/cloud versions.
if not defined SYNCWB_EXPORT_OUTPUT set "SYNCWB_EXPORT_OUTPUT=%USERPROFILE%\Documents\SyncWB\training_exports\living_lab_09_SY_19_MM_initial_v001_v3"
if not defined SYNCWB_KINECT_ROOT set "SYNCWB_KINECT_ROOT=D:\smart_cup_recordings\Kinect"
set "SYNCWB_SPATIAL_CALIBRATION=%~dp0..\..\calibration\kinect_radar\2026-10-06-desk\desk_all.json"
set "SYNCWB_EXPORT_SELECTION=%~dp0living_lab_09_SY_19_MM_selection.json"
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
    echo Cannot find spatial calibration: "%SYNCWB_SPATIAL_CALIBRATION%"
    goto failed
)
call "%SYNCWB_CONDA_EXE%" run --no-capture-output --prefix "%SYNCWB_SOURCE_ENV%" python -m sync_workbench.cli.main export-training-data ^
  --sqlite "%CD%\workbench.sqlite" ^
  --artifact-root "%CD%\artifact_store" ^
  --selection "%SYNCWB_EXPORT_SELECTION%" ^
  --kinect-root "%SYNCWB_KINECT_ROOT%" ^
  --spatial-calibration "%SYNCWB_SPATIAL_CALIBRATION%" ^
  --read-only-root "D:\smart_cup_recordings" ^
  --output "%SYNCWB_EXPORT_OUTPUT%" %*
set "SYNCWB_EXIT_CODE=%ERRORLEVEL%"
if not "%SYNCWB_EXIT_CODE%"=="0" if not defined SYNCWB_NO_PAUSE pause
popd
endlocal & exit /b %SYNCWB_EXIT_CODE%
:failed
if not defined SYNCWB_NO_PAUSE pause
popd
exit /b 1
