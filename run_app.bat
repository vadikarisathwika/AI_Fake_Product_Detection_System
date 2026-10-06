@echo off
echo Installing required packages...
python -m pip install -r requirements.txt
echo.
echo Starting AI Fake Product Detection System...
streamlit run app.py
pause
