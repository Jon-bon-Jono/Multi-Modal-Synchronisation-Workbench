@echo off
setlocal
pushd "%~dp0..\.." || exit /b 1

syncwb build-artifacts ^
  --input-temp "C:/Users/GSBME/SmartCupStudy/Unified_network/data_sets/UNSW-PANOPTES/UNSW-PANOPTES-ETL-Pipeline/docs/sync_workbench/temp_ingestion_package" ^
  --sqlite workbench.sqlite ^
  --artifact-root artifact_store ^
  --subject 07_SW ^
  --overwrite

set "SYNCWB_EXIT_CODE=%ERRORLEVEL%"
popd
endlocal & exit /b %SYNCWB_EXIT_CODE%
