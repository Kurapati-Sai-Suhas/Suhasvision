@echo off
REM Runs every automated test suite in the repo, each from the working
REM directory it requires, using the shared backend\venv environment.
REM Exits non-zero if any suite fails.
REM
REM Deliberately EXCLUDED: dataset\test_phase2.py. Despite the name it is not
REM a unittest file -- it is a manual integration script that appends mock
REM rows to the real anonymized_inference_tensors.csv / coach_overrides.csv
REM and launches a full retraining run. Only run it on purpose, by hand.

setlocal
set REPO=%~dp0
set PY=%REPO%backend\venv\Scripts\python.exe
set FAILED=0

echo ============================================
echo  dataset unit tests
echo ============================================
pushd "%REPO%dataset"
"%PY%" -m unittest test_pipeline_logic test_zero_storage_pipeline test_model_layers test_inference_service test_audit_duplicates test_subject_selection
if errorlevel 1 set FAILED=1
popd

echo.
echo ============================================
echo  academic_scripts unit tests
echo ============================================
pushd "%REPO%dataset\academic_scripts"
"%PY%" -m unittest test_feature_engineering test_stance_symmetry_confidence test_evaluation_protocol test_ablation_stance_symmetry_filter
if errorlevel 1 set FAILED=1
popd

echo.
echo ============================================
echo  Django API tests
echo ============================================
pushd "%REPO%backend\backend"
"%PY%" manage.py test api
if errorlevel 1 set FAILED=1
popd

echo.
if %FAILED%==1 (
    echo RESULT: FAILURES DETECTED
    exit /b 1
)
echo RESULT: ALL SUITES PASSED
exit /b 0
