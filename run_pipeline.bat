@echo off
cd /d "%~dp0"
echo ============================================== >> run_history.log
echo [%date% %time%] Pipeline run started >> run_history.log
python discover_and_outreach.py --send --non-interactive >> run_history.log 2>&1
echo [%date% %time%] Pipeline run finished >> run_history.log
echo ============================================== >> run_history.log
