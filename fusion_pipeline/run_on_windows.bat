@echo off
REM 在 Windows 环境快速运行示例的脚本

echo ========================================================================================================
echo 算子融合分析流水线 - 快速示例
echo ========================================================================================================
echo.

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 未安装或未添加到 PATH
    pause
    exit /b 1
)

echo [INFO] 运行示例脚本...
echo.

REM 运行示例
python fusion_pipeline\run_example.py

if errorlevel 1 (
    echo.
    echo [ERROR] 运行失败
    pause
    exit /b 1
)

echo.
echo ========================================================================================================
echo [SUCCESS] 分析完成！
echo ========================================================================================================
echo.
echo 输出文件:
echo   - example_output\example_report.txt
echo   - example_output\example_analysis.png
echo.
echo 查看报告:
echo   type example_output\example_report.txt
echo.

pause