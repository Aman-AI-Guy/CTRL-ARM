# Operator re-arm for Trova (docs/TABLE_SCAN_V2.md step 4). Run ONLY by the operator, who by running it confirms:
#   - the robot area is clear (rest-area hard rule) and the E-stop is in hand
#   - the controller is ready (no fault on the robot / Studio)
#   - first run only: no tray lift / fork was ever used on this station (clears the missing lift record)
# Reads TROVA_TOKEN, TROVA_OPERATOR_NAME, TROVA_OPERATOR_TOKEN from .env. Sends no motion command.
param([string]$Repo = "C:\Users\ferra\Desktop\Finale\Grok_Hackathon", [string]$Base = "http://localhost:8000")
$ErrorActionPreference = "Stop"

function EnvValue($name) {
    $line = Get-Content (Join-Path $Repo ".env") | Where-Object { $_ -match "^\s*$name\s*=" } | Select-Object -Last 1
    if (-not $line) { return "" }
    return (($line -split "=", 2)[1] -replace "\s+#.*$", "").Trim()
}
$api = EnvValue "TROVA_TOKEN"
$opName = EnvValue "TROVA_OPERATOR_NAME"
$opToken = EnvValue "TROVA_OPERATOR_TOKEN"
if (-not $opName -or -not $opToken) { throw "Set TROVA_OPERATOR_NAME and TROVA_OPERATOR_TOKEN in .env, then restart the server" }
$h = @{ "x-trova-token" = $api; "x-trova-operator-token" = $opToken }

$s = Invoke-RestMethod "$Base/api/state" -Headers $h
if ($s.arm.motion_mode -ne "hardware") { throw "motion_mode is '$($s.arm.motion_mode)', not hardware: fix the .env unlocks and restart" }

"1/3 clear_errors (enables the controller, no motion)"
Invoke-RestMethod -Method Post "$Base/api/arm/clear_errors" -Headers $h | Out-Null

$s = Invoke-RestMethod "$Base/api/state" -Headers $h
if ($s.lift_interrupted) {
    "2/3 recover_lift (no lift record on this PC yet; operator confirms the fork is free / never used)"
    $body = @{ operator = $opName; stop_nonce = $s.stop_nonce; area_clear = $true; controller_ready = $true
               fork_freed = $true; note = "No tray lift ever run on this station (TROVA_ALLOW_LIFT=0, no fork); first lift record on Windows" }
    Invoke-RestMethod -Method Post "$Base/api/arm/recover_lift" -Headers $h -ContentType "application/json" -Body ($body | ConvertTo-Json) | Out-Null
} else { "2/3 lift record clear, skipped" }

"3/3 reset (software re-arm)"
$s = Invoke-RestMethod "$Base/api/state" -Headers $h
$body = @{ operator = $opName; stop_nonce = $s.stop_nonce; area_clear = $true; controller_ready = $true }
Invoke-RestMethod -Method Post "$Base/api/arm/reset" -Headers $h -ContentType "application/json" -Body ($body | ConvertTo-Json) | Out-Null

$s = Invoke-RestMethod "$Base/api/state" -Headers $h
"motion_mode=$($s.arm.motion_mode) stop_latched=$($s.stop_latched) lift_interrupted=$($s.lift_interrupted) blocked=$($s.arm_blocked)"
