param(
    [ValidateSet("prepare", "plan", "smoke", "core", "temporal", "report", "test")]
    [string]$Action = "plan",
    [int]$Job = -1,
    [int]$MaxJobs = 0,
    [string]$Config = ""
)
$ErrorActionPreference = "Stop"
$studyPython = Join-Path $env:USERPROFILE "anaconda3\envs\syncwb\python.exe"
if (-not (Test-Path -LiteralPath $studyPython)) {
    $studyPython = (Get-Command python -ErrorAction Stop).Source
}
$entry = Join-Path $PSScriptRoot "run_study.py"
$studyArgs = @("-B", $entry)
if ($Config) { $studyArgs += @("--config", $Config) }
switch ($Action) {
    "test" {
        & $studyPython -B -m pytest (Join-Path $PSScriptRoot "tests") -q -p no:cacheprovider
        exit $LASTEXITCODE
    }
    "smoke" { $studyArgs += @("train", "--suite", "smoke") }
    "core" { $studyArgs += @("train", "--suite", "core") }
    "temporal" { $studyArgs += @("train", "--suite", "temporal") }
    default { $studyArgs += $Action }
}
if ($Action -in @("smoke", "core", "temporal")) {
    if ($Job -ge 0) { $studyArgs += @("--job", "$Job") }
    if ($MaxJobs -gt 0) { $studyArgs += @("--max-jobs", "$MaxJobs") }
}
& $studyPython @studyArgs
exit $LASTEXITCODE
