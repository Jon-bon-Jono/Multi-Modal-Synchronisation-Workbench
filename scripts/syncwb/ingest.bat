@echo off
setlocal
pushd "%~dp0..\.." || exit /b 1

syncwb ingest-temp ^
  --input "C:/Users/GSBME/SmartCupStudy/Unified_network/data_sets/UNSW-PANOPTES/UNSW-PANOPTES-ETL-Pipeline/docs/sync_workbench/temp_ingestion_package" ^
  --sqlite workbench.sqlite ^
  --parquet canonical_export ^
  --reports reports

set "SYNCWB_EXIT_CODE=%ERRORLEVEL%"
popd
endlocal & exit /b %SYNCWB_EXIT_CODE%
