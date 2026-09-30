"""Execute generated scripts with a controlled Conda double, never a GUI."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from sync_workbench.deployment.conda_launchers import write_conda_launchers


@pytest.fixture(params=['windows','macos'])
def scripts(request,tmp_path):
    platform=request.param
    if platform=='windows' and os.name!='nt':pytest.skip('Windows shell required')
    bash=shutil.which('bash')
    if not bash and Path('C:/Program Files/Git/bin/bash.exe').exists():bash='C:/Program Files/Git/bin/bash.exe'
    if platform=='macos' and not bash:pytest.skip('Bash required for macOS script logic')
    root=tmp_path/'package with spaces'
    root.mkdir()
    write_conda_launchers(root,'test-runtime')
    stub=tmp_path/('fake conda.cmd' if platform=='windows' else 'fake conda')
    if platform=='windows':
        stub.write_bytes(b'''@echo off\r
echo %*>>"%CONDA_TEST_LOG%"\r
echo %PYTHONPATH%>>"%CONDA_TEST_SOURCE%"
if "%1"=="run" if defined SYNCWB_TEST_EXECUTE (
    "%SYNCWB_TEST_PYTHON%" -c "import syncwb_source_probe as p; print(p.VALUE)"
    exit /b
)
if "%~6"=="-c" if defined SYNCWB_TEST_FAIL_HEALTH exit /b 9
if "%SYNCWB_TEST_FAIL%"=="%1" exit /b 9\r
if "%1"=="create" (\r
    mkdir "%~4\\conda-meta"\r
    echo created>"%~4\\conda-meta\\history"\r
)\r
exit /b 0\r
''')
    else:
        stub.write_bytes('''#!/bin/bash
printf '%s\\n' "$*" >> "$CONDA_TEST_LOG"
printf '%s\n' "$PYTHONPATH" >> "$CONDA_TEST_SOURCE"
if [[ "$1" == run && -n "${SYNCWB_TEST_EXECUTE:-}" ]]; then
    "$SYNCWB_TEST_PYTHON" -c 'import syncwb_source_probe as p; print(p.VALUE)'
    exit $?
fi
if [[ "${6:-}" == -c && -n "${SYNCWB_TEST_FAIL_HEALTH:-}" ]]; then exit 9; fi
if [[ "${SYNCWB_TEST_FAIL:-}" == "$1" ]]; then exit 9; fi
if [[ "$1" == create ]]; then
    "$SYNCWB_TEST_PYTHON" -c 'from pathlib import Path; import sys; p=Path(sys.argv[1])/"conda-meta"; p.mkdir(parents=True,exist_ok=True); (p/"history").write_text("created")' "$4"
fi
'''.encode('utf-8'))
        stub.chmod(0o755)
    env=dict(os.environ,SYNCWB_CONDA_EXE=stub.as_posix(),SYNCWB_ENV_ROOT=(tmp_path/'environment root').as_posix(),
             SYNCWB_NO_PAUSE='1',CONDA_TEST_SOURCE=(tmp_path/'source paths.txt').as_posix(),CONDA_TEST_LOG=(tmp_path/'conda calls.txt').as_posix(),SYNCWB_TEST_PYTHON=sys.executable)
    def run(action,*args):
        file=root/(f'{action}_windows.cmd' if platform=='windows' else f'{action}_macos.command')
        cmd=[os.environ.get('COMSPEC','cmd.exe'),'/d','/c',str(file),*args] if platform=='windows' else [bash,'-c','export PATH="/usr/bin:/bin:$PATH"; exec /bin/bash "$@"','syncwb-test',str(file),*args]
        result=subprocess.run(cmd,cwd=tmp_path,env=env,capture_output=True,text=True,timeout=30)
        if result.returncode:print(result.stdout,result.stderr)
        return result
    return root,env,run


def test_setup_then_launch_and_resume_with_spaces(scripts):
    root,env,run=scripts
    result=run('setup')
    assert result.returncode==0,(result.stdout,result.stderr)
    for step in ('[1/4]','[2/4]','[3/4]','[4/4]','Setup complete'):assert step in result.stdout
    calls=Path(env['CONDA_TEST_LOG']).read_text()
    assert 'create --yes --prefix' in calls and 'python=3.11 pip' in calls
    assert '--no-capture-output' in calls and '-m pip install' in calls
    assert '--verify-only --full-checksums --sqlite-integrity-check' in calls
    launched=run('launch','--verify-only')
    assert launched.returncode==0,(launched.stdout,launched.stderr)
    assert Path(env['CONDA_TEST_LOG']).read_text().splitlines()[-1].endswith('--verify-only')
    resumed=run('setup')
    assert resumed.returncode==0
    calls=Path(env['CONDA_TEST_LOG']).read_text()
    assert calls.count('create --yes')==1 and 'install --yes --prefix' not in calls
    assert 'Reusing existing environment' in resumed.stdout


def test_missing_conda_reports_actionable_error(scripts,tmp_path):
    _,env,run=scripts
    env['SYNCWB_CONDA_EXE']=(tmp_path/'missing conda').as_posix()
    result=run('setup')
    assert result.returncode!=0
    assert 'Cannot find' in result.stdout+result.stderr
    assert 'Setup complete' not in result.stdout
    assert not Path(env['CONDA_TEST_LOG']).exists()


def test_failed_environment_creation_stops_before_install(scripts):
    _,env,run=scripts
    env['SYNCWB_TEST_FAIL']='create'
    result=run('setup')
    assert result.returncode!=0
    assert 'Setup failed' in result.stdout+result.stderr
    assert '[3/4]' not in result.stdout
    assert 'run --' not in Path(env['CONDA_TEST_LOG']).read_text()


def test_failed_install_stops_before_verification(scripts):
    _,env,run=scripts
    env['SYNCWB_TEST_FAIL']='run'
    result=run('setup')
    assert result.returncode!=0
    assert '[4/4]' not in result.stdout
    assert '--verify-only' not in Path(env['CONDA_TEST_LOG']).read_text()


def test_launch_requires_setup_and_preserves_failure(scripts):
    _,env,run=scripts
    result=run('launch')
    assert result.returncode!=0
    assert 'environment is missing' in result.stdout+result.stderr
    assert run('setup').returncode==0
    env['SYNCWB_TEST_FAIL']='run'
    assert run('launch').returncode==9


def test_conda_env_variable_discovery(scripts):
    _,env,run=scripts
    env['CONDA_EXE']=env.pop('SYNCWB_CONDA_EXE')
    result=run('setup')
    assert result.returncode==0,(result.stdout,result.stderr)


def test_generated_files_are_conda_only_and_portable(tmp_path):
    write_conda_launchers(tmp_path,'runtime')
    for file in tmp_path.iterdir():
        content=file.read_text()
        assert '-m venv' not in content and 'py -3.11' not in content
        if file.suffix in {'.command','.sh'}:assert b'\r' not in file.read_bytes()
    assert 'conda-environments' in (tmp_path/'find_environment_windows.cmd').read_text()


@pytest.mark.parametrize('legacy_id',['d361c810a58bc24a9d9b','869708dace69f4d913fa'])
def test_existing_distributed_environment_launches_new_code_without_setup(scripts,legacy_id):
    root,env,run=scripts
    prefix=Path(env['SYNCWB_ENV_ROOT'])/legacy_id
    (prefix/'conda-meta').mkdir(parents=True)
    (prefix/'conda-meta/history').write_text('existing')
    # A new source hash must not require a new environment or a pip reinstall.
    write_conda_launchers(root,'new-gui-runtime')
    assert run('launch','--verify-only').returncode==0
    calls=Path(env['CONDA_TEST_LOG']).read_text()
    assert legacy_id in calls and 'new-gui-runtime' not in calls
    assert 'create ' not in calls and 'install ' not in calls
    paths=Path(env['CONDA_TEST_SOURCE']).read_text().splitlines()
    assert all(_shell_path(p)==root/'application/src' for p in paths)
    assert run('setup').returncode==0
    calls=Path(env['CONDA_TEST_LOG']).read_text()
    assert 'create ' not in calls and 'install --yes' not in calls
    assert '-m pip install' in calls and '--upgrade' not in calls and '--force-reinstall' not in calls
    assert (prefix/'conda-meta/history').read_text()=='existing'


def test_setup_repairs_existing_python_in_place(scripts):
    _,env,run=scripts
    assert run('setup').returncode==0
    env['SYNCWB_TEST_FAIL_HEALTH']='1'
    result=run('setup')
    assert result.returncode==0,(result.stdout,result.stderr)
    assert 'Repairing Python 3.11' in result.stdout
    calls=Path(env['CONDA_TEST_LOG']).read_text()
    assert calls.count('create --yes')==1 and calls.count('install --yes --prefix')==1
    env['SYNCWB_TEST_FAIL']='install'
    assert run('setup').returncode!=0


def test_new_package_uses_its_own_source_with_shared_environment(scripts):
    root,env,run=scripts
    assert run('setup').returncode==0
    assert run('launch').returncode==0
    write_conda_launchers(root,'another-code-hash')
    assert run('launch').returncode==0
    calls=Path(env['CONDA_TEST_LOG']).read_text().splitlines()
    assert all('syncwb-py311' in call for call in calls)
    assert sum('create --yes' in call for call in calls)==1
    assert sum('-m pip install' in call for call in calls)==1
    # Both setup and launch select this package's source, never the installed copy.
    assert all(_shell_path(p)==root/'application/src' for p in Path(env['CONDA_TEST_SOURCE']).read_text().splitlines())
    assert all('application"' not in call for call in calls)


def _shell_path(value):
    # Git Bash reports /c/... while these tests run under Windows Python.
    if os.name=='nt' and len(value)>3 and value[0]=='/' and value[2]=='/':
        value=value[1]+':'+value[2:]
    return Path(value)


def test_launch_imports_updated_bundled_source_without_install(scripts):
    root,env,run=scripts
    assert run('setup').returncode==0
    source=root/'application/src'
    source.mkdir(parents=True)
    module=source/'syncwb_source_probe.py'
    env['SYNCWB_TEST_EXECUTE']='1'
    for value in ('first-package-code','updated-package-code'):
        module.write_text(f'VALUE={value!r}')
        result=run('launch')
        assert result.returncode==0,(result.stdout,result.stderr)
        assert value in result.stdout
    calls=Path(env['CONDA_TEST_LOG']).read_text()
    assert calls.count('create --yes')==1 and calls.count('-m pip install')==1


def test_launch_forwards_optional_verification_flags(scripts):
    _,env,run=scripts
    assert run('setup').returncode==0
    assert run('launch','--verify-only','--full-checksums','--sqlite-integrity-check').returncode==0
    assert Path(env['CONDA_TEST_LOG']).read_text().splitlines()[-1].endswith('--verify-only --full-checksums --sqlite-integrity-check')
    assert run('launch').returncode==0
    last=Path(env['CONDA_TEST_LOG']).read_text().splitlines()[-1]
    assert '--full-checksums' not in last and '--sqlite-integrity-check' not in last
