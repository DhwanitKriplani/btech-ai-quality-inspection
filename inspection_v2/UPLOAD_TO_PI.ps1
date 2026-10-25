param([string]$Target,[string]$Package)
$ErrorActionPreference = 'Stop'
if (-not $Target) { $Target = Read-Host 'Raspberry Pi login, e.g. youruser@192.168.1.50' }
if ($Target -notmatch '^[A-Za-z0-9._-]+@[A-Za-z0-9.-]+$') { throw 'Use user@IPv4-or-hostname; spaces/shell symbols are not accepted.' }
if (-not $Package) { $Package = Join-Path $PSScriptRoot '..\inspection_complete_v2.zip' }
if (-not (Test-Path -LiteralPath $Package -PathType Leaf)) { throw 'ZIP not found. Pass -Package with the actual inspection_complete_v2.zip path.' }
$Package = (Resolve-Path -LiteralPath $Package).Path
Get-Command scp.exe,ssh.exe -ErrorAction Stop | Out-Null
$piRemoteFolder = 'material_inspection_v2_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
& scp.exe -- $Package "$($Target):~/inspection_complete_v2.zip"
if ($LASTEXITCODE -ne 0) { throw 'Upload failed; check Pi hostname/login and network.' }
& ssh.exe -- $Target "mkdir -p ~/$piRemoteFolder && unzip ~/inspection_complete_v2.zip -d ~/$piRemoteFolder"
if ($LASTEXITCODE -ne 0) { throw 'Remote extraction failed. ZIP is uploaded; ensure unzip is installed on Pi.' }
Write-Host "Uploaded. On Pi: cd ~/$piRemoteFolder/inspection_v2"
Write-Host 'Then: bash setup_pi.sh'
Write-Host 'Test bundled samples: bash start_pi.sh demo'
