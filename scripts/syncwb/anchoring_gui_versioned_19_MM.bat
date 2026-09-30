@echo off
setlocal
call "%~dp0run_source_gui.cmd" ^
  --subject 19_MM ^
  --mapping-version initial_rgb_to_raw_v001 ^
  --point-cloud-version raw_d3f274e06a775b757e89dc86f8f57dfb12ab4a198e66912af21dd176d39620f3 ^
  --annotator-id JW01 %*
exit /b %ERRORLEVEL%
