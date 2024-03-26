
set index_url=%1

python setup.py develop --index-url %index_url%

if %errorlevel% neq 0 (
    echo Installing Paqarin failed. Please try again
    exit /b %errorlevel%
)

FOR %%A IN (%*) DO (
    if "%%A"=="/synthcity" (
        echo Installing Synthcity
        python -m pip install synthcity==0.2.4
    )

    if "%%A"=="/ydata" (
        echo Installing YData
        python -m pip install ydata-synthetic==1.3.2 --ignore-requires-python
    )

    if "%%A"=="/sdv" (
        echo Installing SDV
        python -m pip install sdv==1.8.0
    )
)



FOR /f "tokens=*" %%a in (install-requirements.txt) DO (
    echo package=%%a
    if NOT # == %%a (
        python -m pip install %%a
    )
)
python -m pip install --no-deps autogluon.timeseries==1.0.0