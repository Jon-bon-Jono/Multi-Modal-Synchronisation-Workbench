@echo off
setlocal DisableDelayedExpansion
pushd "%~dp0..\.." || exit /b 1
set "PYTHONPATH=%CD%\src"
set "PYTHONNOUSERSITE=1"
set "PYTHONDONTWRITEBYTECODE=1"
rem Calibration is configured inside the calling .bat preset.
set "SYNCWB_CALIBRATION_ARGS="
if defined SYNCWB_SPATIAL_CALIBRATION (
    if not exist "%SYNCWB_SPATIAL_CALIBRATION%" (
        echo Cannot find the preset spatial calibration: "%SYNCWB_SPATIAL_CALIBRATION%"
        goto failed
    )
    set SYNCWB_CALIBRATION_ARGS=--spatial-calibration "%SYNCWB_SPATIAL_CALIBRATION%"
)
rem The selected subject/run resolves its video through the database in the GUI.
if not defined SYNCWB_RGB_ROOT if exist "D:\smart_cup_recordings\Kinect\" set "SYNCWB_RGB_ROOT=D:\smart_cup_recordings\Kinect"
if not defined SYNCWB_RGB_ROOT set "SYNCWB_RGB_ROOT=%USERPROFILE%\Documents\SyncWB\backend_validation\rgb_root"
if not exist "%SYNCWB_RGB_ROOT%\" (
    echo Cannot find the Kinect RGB root. Set SYNCWB_RGB_ROOT to its Kinect root and retry.
    goto failed
)
if not defined SYNCWB_CONDA_EXE set "SYNCWB_CONDA_EXE=%USERPROFILE%\anaconda3\Scripts\conda.exe"
if not defined SYNCWB_SOURCE_ENV set "SYNCWB_SOURCE_ENV=%USERPROFILE%\anaconda3\envs\syncwb"
if not exist "%SYNCWB_CONDA_EXE%" (
    echo Cannot find Conda. Set SYNCWB_CONDA_EXE to its executable path and retry.
    goto failed
)
echo Source database: "%CD%\workbench.sqlite"
echo Artifact root: "%CD%\artifact_store"
echo RGB root: "%SYNCWB_RGB_ROOT%"
call "%SYNCWB_CONDA_EXE%" run --no-capture-output --prefix "%SYNCWB_SOURCE_ENV%" python -m sync_workbench.cli.main anchoring-gui --sqlite "%CD%\workbench.sqlite" --artifact-root "%CD%\artifact_store" --rgb-root "%SYNCWB_RGB_ROOT%" %* %SYNCWB_CALIBRATION_ARGS%
set "SYNCWB_EXIT_CODE=%ERRORLEVEL%"
if not "%SYNCWB_EXIT_CODE%"=="0" if not defined SYNCWB_NO_PAUSE pause
popd
endlocal & exit /b %SYNCWB_EXIT_CODE%
:failed
if not defined SYNCWB_NO_PAUSE pause
popd
exit /b 1
