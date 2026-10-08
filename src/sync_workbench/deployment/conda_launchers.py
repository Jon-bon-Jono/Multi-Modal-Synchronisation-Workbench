"""Portable Conda setup/launch scripts with streamed progress and explicit errors."""
from pathlib import Path

WINDOWS_FIND = r'''@echo off
set "CONDA_CMD="
if defined SYNCWB_CONDA_EXE (
    if not exist "%SYNCWB_CONDA_EXE%" (
        echo Cannot find Conda at SYNCWB_CONDA_EXE: "%SYNCWB_CONDA_EXE%"
        exit /b 1
    )
    set "CONDA_CMD=%SYNCWB_CONDA_EXE%"
    exit /b 0
)
if defined CONDA_EXE if exist "%CONDA_EXE%" set "CONDA_CMD=%CONDA_EXE%"
if defined CONDA_CMD exit /b 0
for /f "delims=" %%C in ('where conda.exe 2^>nul') do if not defined CONDA_CMD set "CONDA_CMD=%%C"
for /f "delims=" %%C in ('where conda.bat 2^>nul') do if not defined CONDA_CMD set "CONDA_CMD=%%C"
if defined CONDA_CMD exit /b 0
for %%D in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%USERPROFILE%\miniforge3" "%USERPROFILE%\mambaforge" "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\miniconda3" "%LOCALAPPDATA%\miniforge3" "%ProgramData%\anaconda3" "%ProgramData%\miniconda3" "%ProgramData%\miniforge3") do if not defined CONDA_CMD if exist "%%~D\Scripts\conda.exe" set "CONDA_CMD=%%~D\Scripts\conda.exe"
if defined CONDA_CMD exit /b 0
echo Conda was not found. Install Anaconda, Miniconda or Miniforge, then retry.
echo For a custom installation, run this script from Anaconda Prompt, or set SYNCWB_CONDA_EXE to its conda.exe path.
exit /b 1
'''

MAC_FIND = r'''# Sourced by the setup/launch scripts; no shell initialization required.
CONDA_CMD=""
if [[ -n "${SYNCWB_CONDA_EXE:-}" ]]; then
    if [[ ! -x "$SYNCWB_CONDA_EXE" ]]; then
        echo "Cannot find executable Conda at SYNCWB_CONDA_EXE: $SYNCWB_CONDA_EXE" >&2
        return 1
    fi
    CONDA_CMD="$SYNCWB_CONDA_EXE"
elif [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then
    CONDA_CMD="$CONDA_EXE"
else
    CONDA_CMD="$(type -P conda || true)"
    if [[ -z "$CONDA_CMD" ]]; then
        for prefix in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/mambaforge" /opt/anaconda3 /opt/miniconda3 /opt/miniforge3 /usr/local/anaconda3 /usr/local/miniconda3 /usr/local/miniforge3; do
            if [[ -x "$prefix/bin/conda" ]]; then CONDA_CMD="$prefix/bin/conda"; break; fi
        done
    fi
fi
if [[ -z "$CONDA_CMD" ]]; then
    echo "Conda was not found. Install Anaconda, Miniconda or Miniforge, then retry." >&2
    echo "For a custom installation, set SYNCWB_CONDA_EXE to its bin/conda path." >&2
    return 1
fi
'''

WINDOWS_ENVIRONMENT = r'''@echo off
set "ENV_ROOT=%LOCALAPPDATA%\SyncWB\conda-environments"
if defined SYNCWB_ENV_ROOT set "ENV_ROOT=%SYNCWB_ENV_ROOT%"
set "ENV_DIR=%ENV_ROOT%\syncwb-py311"
if exist "%ENV_DIR%\conda-meta\history" exit /b 0
for %%R in (@RUNTIME@ 869708dace69f4d913fa d361c810a58bc24a9d9b) do if exist "%ENV_ROOT%\%%R\conda-meta\history" (
    set "ENV_DIR=%ENV_ROOT%\%%R"
    exit /b 0
)
exit /b 0
'''

MAC_ENVIRONMENT = r'''# Stable dependency environment; reuse either previously distributed environment.
ENV_ROOT="${SYNCWB_ENV_ROOT:-$HOME/Library/Application Support/SyncWB/conda-environments}"
ENV_DIR="$ENV_ROOT/syncwb-py311"
if [[ ! -f "$ENV_DIR/conda-meta/history" ]]; then
    for runtime in @RUNTIME@ 869708dace69f4d913fa d361c810a58bc24a9d9b; do
        if [[ -f "$ENV_ROOT/$runtime/conda-meta/history" ]]; then
            ENV_DIR="$ENV_ROOT/$runtime"
            break
        fi
    done
fi
'''

WINDOWS_SETUP = r'''@echo off
setlocal DisableDelayedExpansion
call "%~dp0find_environment_windows.cmd"
set "PYTHONPATH=%~dp0application\src"
set "PYTHONNOUSERSITE=1"
set "PYTHONDONTWRITEBYTECODE=1"
echo [1/4] Finding Conda...
call "%~dp0find_conda_windows.cmd" || goto failed
echo Conda: "%CONDA_CMD%"
echo [2/4] Preparing Python 3.11 environment at "%ENV_DIR%"...
if not exist "%ENV_DIR%\conda-meta\history" goto create_environment
echo Reusing existing environment. Checking Python and pip...
call "%CONDA_CMD%" run --no-capture-output --prefix "%ENV_DIR%" python -c "import sys, pip; sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)"
if not errorlevel 1 goto dependencies
echo Repairing Python 3.11 and pip in the existing environment...
call "%CONDA_CMD%" install --yes --prefix "%ENV_DIR%" python=3.11 pip
if errorlevel 1 goto failed
goto dependencies
:create_environment
echo Creating environment. Conda download and installation output follows...
call "%CONDA_CMD%" create --yes --prefix "%ENV_DIR%" python=3.11 pip
if errorlevel 1 goto failed
:dependencies
echo [3/4] Checking dependencies; installing only missing or incompatible packages...
call "%CONDA_CMD%" run --no-capture-output --prefix "%ENV_DIR%" python -u -m pip install --progress-bar on -r "%~dp0requirements.txt"
if errorlevel 1 goto failed
echo [4/4] Verifying full package checksums and SQLite integrity. This setup check may take a little time...
call "%CONDA_CMD%" run --no-capture-output --prefix "%ENV_DIR%" python -u -m sync_workbench.deployment.student_runtime --package "%~dp0." --verify-only --full-checksums --sqlite-integrity-check
if errorlevel 1 goto failed
echo Setup complete. Run launch_windows.cmd.
if not defined SYNCWB_NO_PAUSE pause
exit /b 0
:failed
echo Setup failed. Read the error above. Fix it, then rerun setup_windows.cmd.
echo No existing student work has been replaced.
if not defined SYNCWB_NO_PAUSE pause
exit /b 1
'''

WINDOWS_LAUNCH = r'''@echo off
setlocal DisableDelayedExpansion
call "%~dp0find_environment_windows.cmd"
set "PYTHONPATH=%~dp0application\src"
set "PYTHONNOUSERSITE=1"
set "PYTHONDONTWRITEBYTECODE=1"
echo Finding Conda...
call "%~dp0find_conda_windows.cmd" || goto failed
if not exist "%ENV_DIR%\conda-meta\history" (
    echo This package's environment is missing. Run setup_windows.cmd first.
    goto failed
)
echo Checking package identity and opening SyncWB...
call "%CONDA_CMD%" run --no-capture-output --prefix "%ENV_DIR%" python -u -m sync_workbench.deployment.student_runtime --package "%~dp0." %*
set "SYNCWB_STATUS=%ERRORLEVEL%"
if not "%SYNCWB_STATUS%"=="0" (
    echo SyncWB did not complete successfully. Read the error above.
    if not defined SYNCWB_NO_PAUSE pause
)
endlocal & exit /b %SYNCWB_STATUS%
:failed
if not defined SYNCWB_NO_PAUSE pause
exit /b 1
'''

MAC_SETUP = r'''#!/bin/bash
set -euo pipefail
trap 'code=$?; echo "Setup failed. Read the error above, fix it, then rerun setup_macos.command." >&2; exit "$code"' ERR
PACKAGE_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
source "$PACKAGE_DIR/find_environment_macos.sh"
export PYTHONPATH="$PACKAGE_DIR/application/src"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
echo "[1/4] Finding Conda..."
source "$PACKAGE_DIR/find_conda_macos.sh"
echo "Conda: $CONDA_CMD"
echo "[2/4] Preparing Python 3.11 environment at $ENV_DIR..."
if [[ -f "$ENV_DIR/conda-meta/history" ]]; then
    echo "Reusing existing environment. Checking Python and pip..."
    if ! "$CONDA_CMD" run --no-capture-output --prefix "$ENV_DIR" python -c "import sys, pip; sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)"; then
        echo "Repairing Python 3.11 and pip in the existing environment..."
        "$CONDA_CMD" install --yes --prefix "$ENV_DIR" python=3.11 pip
    fi
else
    echo "Creating environment. Conda download and installation output follows..."
    "$CONDA_CMD" create --yes --prefix "$ENV_DIR" python=3.11 pip
fi
echo "[3/4] Checking dependencies; installing only missing or incompatible packages..."
"$CONDA_CMD" run --no-capture-output --prefix "$ENV_DIR" python -u -m pip install --progress-bar on -r "$PACKAGE_DIR/requirements.txt"
echo "[4/4] Verifying full package checksums and SQLite integrity. This setup check may take a little time..."
"$CONDA_CMD" run --no-capture-output --prefix "$ENV_DIR" python -u -m sync_workbench.deployment.student_runtime --package "$PACKAGE_DIR" --verify-only --full-checksums --sqlite-integrity-check
echo "Setup complete. Run bash launch_macos.command."
'''

MAC_LAUNCH = r'''#!/bin/bash
set -euo pipefail
trap 'code=$?; echo "SyncWB could not start. Read the error above." >&2; exit "$code"' ERR
PACKAGE_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
source "$PACKAGE_DIR/find_environment_macos.sh"
export PYTHONPATH="$PACKAGE_DIR/application/src"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
echo "Finding Conda..."
source "$PACKAGE_DIR/find_conda_macos.sh"
if [[ ! -f "$ENV_DIR/conda-meta/history" ]]; then
    echo "This package's environment is missing. Run bash setup_macos.command first." >&2
    exit 1
fi
echo "Checking package identity and opening SyncWB..."
"$CONDA_CMD" run --no-capture-output --prefix "$ENV_DIR" python -u -m sync_workbench.deployment.student_runtime --package "$PACKAGE_DIR" "$@"
'''

SETUP = '''Student package - Conda installation and launch

Requires an existing Anaconda, Miniconda or Miniforge installation and internet
for initial setup. A separate Python installation or Python launcher is not needed.
Extract the WHOLE folder into a writable local directory before running setup.

Windows: run setup_windows.cmd, then launch_windows.cmd.
macOS: in Terminal, cd into this folder and run bash setup_macos.command,
then bash launch_macos.command.

If config.json includes spatial_calibration, launch automatically uses that
bundled Kinect/raw-radar calibration. No calibration argument or student file
selection is needed. Details shows its filename and checksum. Do not edit the
bundled calibration/configuration; request a revised package from the coordinator.

Setup finds Conda, reuses an existing SyncWB Python 3.11 environment or creates one,
installs only missing/incompatible dependencies, then verifies the package. Each step is announced and Conda/pip output is visible.
Setup can be rerun after a failure. Follow any Conda channel/account messages shown.
No conda activate or conda init is required. Common installation paths are detected.
If a custom Conda installation is not found, on Windows use Anaconda Prompt, or
set SYNCWB_CONDA_EXE to the full path of conda.exe (Windows) or bin/conda (macOS).
Do not edit these scripts: all shipped files are checked against the manifest.

The dedicated Conda environment is outside the package in your user profile.
GUI/code-only updates: extract the new package and run its launch script directly.
The launcher uses code from that package and reuses the dependency environment.
Both earlier Conda student packages are recognized automatically.
Dependency updates or incomplete setup: rerun setup. It updates the same environment,
keeps satisfying dependencies, and does not recreate it or upgrade everything.
Healthy existing Python/pip skips the Conda solve/download step entirely.
Moving the whole package on the same computer does not require reinstalling;
a different computer needs setup unless it already has a SyncWB environment.
Setup does not change your base environment or any student work.
Actual macOS installation and graphics compatibility remain part of WP3.

On first launch enter your assigned annotator ID (letters/numbers/._- only).
It is remembered in work/annotator.json. Give each student a fresh copy.
The working database is work/workbench.sqlite; do not delete the work folder.
Use Export anchors to save a return JSON. Recovery snapshots are in work/recovery.
Keep your entire work folder until the coordinator has accepted your returned work.
Normal launch uses lightweight identity checks; it does not scan asset contents or
hash database rows. For optional full verification without opening the GUI:
Windows: launch_windows.cmd --verify-only --full-checksums --sqlite-integrity-check
macOS: bash launch_macos.command --verify-only --full-checksums --sqlite-integrity-check
Either check flag can be used alone. Omit --verify-only to open the GUI afterward.
Setup always requests both checks. Full student instructions follow in WP4.

This is a newly built package. Extract into a NEW folder. Keep any old package/work
folder containing annotations; do not move its database into this new package.
'''


def write_conda_launchers(root, runtime_id):
    files = {'setup_windows.cmd':WINDOWS_SETUP,'launch_windows.cmd':WINDOWS_LAUNCH,
             'setup_macos.command':MAC_SETUP,'launch_macos.command':MAC_LAUNCH,
             'find_conda_windows.cmd':WINDOWS_FIND,'find_conda_macos.sh':MAC_FIND,
             'find_environment_windows.cmd':WINDOWS_ENVIRONMENT,
             'find_environment_macos.sh':MAC_ENVIRONMENT,'SETUP.txt':SETUP}
    for name, content in files.items():
        content = content.replace('@RUNTIME@',runtime_id)
        if name.endswith('.cmd'):
            content = content.replace('\n','\r\n')
        (Path(root)/name).write_bytes(content.encode('utf-8'))
