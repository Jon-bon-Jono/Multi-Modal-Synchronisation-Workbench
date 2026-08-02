@echo off
setlocal
pushd "%~dp0..\.." || exit /b 1

python ml\mmfi_syncwb_domain_diagnostics.py ^
  --packed-root "C:\Users\GSBME\SmartCupStudy\Unified_network\data_sets\MM-Fi\packed_data" ^
  --sqlite "workbench.sqlite" ^
  --artifact-root "artifact_store" ^
  --subject "19_MM" ^
  --mapping-version "piecewise_rgb_to_pc_v001_map" ^
  --window-size 5 ^
  --out "runs\domain_diagnostics_19_MM_corrected_signal"

set "SYNCWB_EXIT_CODE=%ERRORLEVEL%"
popd
endlocal & exit /b %SYNCWB_EXIT_CODE%
