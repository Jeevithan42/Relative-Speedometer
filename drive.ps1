#requires -version 5
<#
  Test-drive launcher for the Microsoft LifeCam (external USB).

  Defaults to device index 1 -- index 0 is the built-in laptop camera (HANDOFF.md 2).
  Confirm the index with:  python main.py cameras --probe

  A wrong index does NOT produce an error. On this machine index 1 with nothing plugged
  in opens fine and delivers 1280x720 frames of near-black, so the pipeline runs on a
  dead feed, detects nothing, and says nothing. The pre-flight below reports brightness
  for that reason -- it is the only thing that tells a camera from a phantom.

  Recording goes to recordings\drive-<timestamp>.mp4. -Fps sets only the playback rate
  of that file; the live math timestamps every frame on arrival from the wall clock.

  In the window:  Q quits, and quitting is what flushes the recording.
#>
param(
    [int]$Index      = 1,
    [int]$Width      = 1280,
    [int]$Height     = 720,
    [int]$Stride     = 5,
    [int]$InputSize  = 416,
    [double]$Fps     = 10,   # playback rate of the recording only; measured 9.4 with -Duration 12
    [double]$Duration = 0,   # seconds; 0 = run until you press Q
    [int]$PreflightTimeout = 40,
    [switch]$NoSave,
    [switch]$SkipPreflight,
    [switch]$PreflightOnly,
    [switch]$SkipBrightnessCheck
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

if (-not $SkipPreflight) {
    Write-Host "pre-flight: does index $Index deliver frames at ${Width}x${Height}?"
    $log = Join-Path $env:TEMP ("rspeed-preflight-{0}.txt" -f $PID)
    $p = Start-Process -FilePath 'python' `
                       -ArgumentList @('tools/probe_index.py', $Index, $Width, $Height) `
                       -NoNewWindow -PassThru `
                       -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    # WaitForExit($ms), not Wait-Process: the latter leaves ExitCode unpopulated on the
    # object Start-Process hands back, so a successful probe reads as a failure.
    $exited = $p.WaitForExit($PreflightTimeout * 1000)
    if (-not $exited) {
        try { Stop-Process -Id $p.Id -Force -ErrorAction Stop } catch {}
        Write-Host ''
        Write-Host "index $Index never delivered a frame ($PreflightTimeout s)." -ForegroundColor Red
        Write-Host 'That is the phantom-device case: nothing is plugged in at that index.'
        Write-Host 'Plug the LifeCam in, then: python main.py cameras --probe'
        Write-Host 'and re-run with the index it reports, e.g. .\drive.ps1 -Index 2'
        exit 1
    }
    $out = ''
    if (Test-Path -LiteralPath $log) {
        $raw = Get-Content -Raw -LiteralPath $log
        if ($raw) { $out = $raw.Trim() }
    }
    # Judge on what the probe printed. Its "ok <w>x<h> <brightness>" line is the contract.
    if ($out -notmatch '^ok\s') {
        Write-Host ''
        Write-Host "index $Index is not usable." -ForegroundColor Red
        if ($out) { Write-Host $out }
        if (Test-Path -LiteralPath "$log.err") { Write-Host (Get-Content -Raw -LiteralPath "$log.err").Trim() }
        Write-Host 'Run: python main.py cameras --probe'
        exit 1
    }
    Write-Host "pre-flight $out" -ForegroundColor Green
    $fields = $out -split '\s+'
    if ($fields.Count -ge 2 -and $fields[1] -ne "${Width}x${Height}") {
        Write-Host "note: camera delivered $($fields[1]), not ${Width}x${Height}." -ForegroundColor Yellow
        Write-Host 'calibration.json is fx=981px at 1280x720. A different aspect ratio is refused;'
        Write-Host 'a same-aspect mode is scaled. Metres depend on this being right.'
    }
    if ($fields.Count -ge 3 -and [double]$fields[2] -lt 10) {
        Write-Host ''
        Write-Host "STOP: mean brightness $($fields[2]) is a black frame." -ForegroundColor Red
        Write-Host 'That is the signature of an index with no camera on it: DSHOW opens a'
        Write-Host 'phantom device and hands back black frames, and nothing downstream'
        Write-Host 'complains -- every quality gate just rejects frames with no gradients.'
        Write-Host 'Check the LifeCam is plugged in, the lens cap is off, and the index is'
        Write-Host 'right: python main.py cameras --probe'
        Write-Host 'A real daytime road scene reads 40+. Re-run with -SkipBrightnessCheck'
        Write-Host 'if you genuinely mean to run in the dark.' -ForegroundColor Yellow
        if (-not $SkipBrightnessCheck) { exit 1 }
    }
}

if ($PreflightOnly) {
    Write-Host 'pre-flight only; not launching.'
    exit 0
}

$cmd = @('main.py', 'video',
         '--source', $Index,
         '--width', $Width, '--height', $Height,
         '--input-size', $InputSize,
         '--detect-stride', $Stride,
         '--show')

if ($Duration -gt 0) { $cmd += @('--duration', $Duration) }

# noise.json, if measure-noise has been run on this camera, is picked up automatically.
# Without it the config placeholders are used, and on the LifeCam those are optimistic --
# sigma_s by 2.2x (HANDOFF.md 4). Fall back to the values measured on this camera.
if (Test-Path -LiteralPath 'noise.json') {
    Write-Host 'noise.json found - using the measured noise profile.'
} else {
    Write-Host 'no noise.json - using the LifeCam values from HANDOFF.md 4.'
    Write-Host "Run 'python main.py measure-noise --index $Index' once to do better."
    $cmd += @('--sigma-w', '2.830', '--sigma-s', '0.00899')
}

if (-not $NoSave) {
    if (-not (Test-Path -LiteralPath 'recordings')) {
        New-Item -ItemType Directory -Path 'recordings' | Out-Null
    }
    $out = Join-Path 'recordings' ('drive-{0}.mp4' -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
    $cmd += @('--save', $out, '--fps', $Fps)
    Write-Host "recording to $out  (press Q in the window to flush it)"
}

Write-Host ''
Write-Host "python $($cmd -join ' ')" -ForegroundColor Cyan
Write-Host ''
& python @cmd
