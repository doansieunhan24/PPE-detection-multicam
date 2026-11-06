# Live demo on C2 s01 recordings played as RTSP cameras (docker/README.md "Demo live").
#   powershell -ExecutionPolicy Bypass -File tools\demo_c2.ps1          # start everything
#   powershell -ExecutionPolicy Bypass -File tools\demo_c2.ps1 -Stop    # stop everything
# Needs data/demo/c2/ (gitignored): raw/cam{1,2,3}.mp4 (takes T04, T02, T03, T08 joined),
# session/{cameras,zones}.yaml, configs/ (copy of configs/ with the s01 CV-tuned P thresholds).
# Opens four windows: RTSP simulator and DeepStream (WSL, tools/demo_c2_wsl.sh), live analytics,
# dashboard; then the browser.
param(
    [string]$Demo = "data/demo/c2",
    [int]$Port = 8765,
    [switch]$Stop
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if ($Stop) {
    wsl -e bash tools/demo_c2_wsl.sh stop
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match 'src\.(analytics_live|dashboard\.app)' } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force; "stopped pid $($_.ProcessId)" }
    return
}

foreach ($p in "$Demo/raw/cam1.mp4", "$Demo/session/cameras.yaml", "$Demo/configs/fusion.yaml") {
    if (-not (Test-Path $p)) { throw "missing $p" }
}
$py = Join-Path $root ".venv\Scripts\python.exe"
$live = "$Demo/live"
if (Test-Path $live) { Remove-Item -Recurse -Force $live }
New-Item -ItemType Directory -Force $live | Out-Null

# 1) simulated cameras rtsp://localhost:8554/cam{1,2,3} (+ path "osd" for the tiled output)
Start-Process wsl -WorkingDirectory $root -ArgumentList "-e bash tools/demo_c2_wsl.sh sim $Demo"
Start-Sleep -Seconds 5
# 2) DeepStream: perception → ZMQ :5555, smart record on alerts from :5556, OSD → rtsp .../osd
Start-Process wsl -WorkingDirectory $root -ArgumentList "-e bash tools/demo_c2_wsl.sh ds $Demo"
# 3) live analytics (fusion P, behavior, alerts, snapshots) and 4) dashboard; logs in $live/
Start-Process $py -WorkingDirectory $root -WindowStyle Minimized `
    -RedirectStandardOutput "$live/analytics.out.log" -RedirectStandardError "$live/analytics.log" `
    -ArgumentList ("-m src.analytics_live --zmq tcp://localhost:5555 " +
    "--sr tcp://localhost:5556 --session $Demo/session --config $Demo/configs --out $live")
Start-Process $py -WorkingDirectory $root -WindowStyle Minimized `
    -RedirectStandardOutput "$live/dashboard.out.log" -RedirectStandardError "$live/dashboard.log" `
    -ArgumentList ("-m src.dashboard.app --results $live " +
    "--session $Demo/session --config $Demo/configs --video-url http://localhost:8889/osd --port $Port")

Start-Sleep -Seconds 25   # DeepStream loads its TensorRT engines
Start-Process "http://localhost:$Port"
"Demo running: dashboard http://localhost:$Port , OSD video http://localhost:8889/osd"
"Stop: powershell -ExecutionPolicy Bypass -File tools\demo_c2.ps1 -Stop"
