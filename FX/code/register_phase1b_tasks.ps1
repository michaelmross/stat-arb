<#
Register the two Phase 1b collector tasks. MUST be run from an ELEVATED
PowerShell: an S4U task (runs whether or not you are logged on, no console
window) can only be registered with administrator rights.

    powershell -ExecutionPolicy Bypass -File register_phase1b_tasks.ps1 -DisableCensusTask
    powershell -ExecutionPolicy Bypass -File register_phase1b_tasks.ps1 -DryRun

Both tasks are cloned from 'FX StatArb Collector' so every hard-won setting
carries over unchanged: S4U logon, 03:00 ET start with a 15-minute retrigger
for 14 h (self-heals after a reboot or an external kill), IgnoreNew so a
retrigger never starts a second collector, restart-on-failure 10 x 5 min, and
a 15 h execution limit. Only the name, the start date and the action differ.

-DisableCensusTask  also disables 'FX StatArb Collector'. The EUR/GBP census
                    ended 2026-09-11; left enabled it would stream on account
                    -002 alongside the AUD/NZD arm, and OANDA may drop one of
                    two streams on the same account. Disabling is reversible:
                    Enable-ScheduledTask -TaskName 'FX StatArb Collector'
-DryRun             print what would be registered; change nothing.
#>
param([switch]$DisableCensusTask, [switch]$DryRun)
$ErrorActionPreference = 'Stop'

$base = (Export-ScheduledTask -TaskName 'FX StatArb Collector')
$base = $base -replace '<Date>[^<]*</Date>', '<Date>2026-09-12T12:00:00</Date>'
$base = $base -replace '<StartBoundary>[^<]*</StartBoundary>', '<StartBoundary>2026-09-14T03:00:00-04:00</StartBoundary>'

foreach ($arm in @('audnzd', 'eurczk')) {
    $name = 'FX StatArb Phase1b ' + $arm.ToUpper()
    $x = $base -replace '<URI>[^<]*</URI>', ('<URI>\' + $name + '</URI>')
    $x = $x -replace 'launch_collector\.vbs"', ('launch_arm.vbs" run_collector_' + $arm + '.bat')
    if ($DryRun) {
        Write-Output "--- $name"
        ($x -split "`n") | Where-Object { $_ -match 'URI|StartBoundary|LogonType|Arguments|Interval|MultipleInstances' } |
            ForEach-Object { Write-Output $_.Trim() }
        continue
    }
    Register-ScheduledTask -TaskName $name -Xml $x -Force | Out-Null
    $i = Get-ScheduledTaskInfo -TaskName $name
    Write-Output ("registered  {0,-28} next run {1}" -f $name, $i.NextRunTime)
}

if ($DisableCensusTask -and -not $DryRun) {
    Disable-ScheduledTask -TaskName 'FX StatArb Collector' | Out-Null
    Write-Output "disabled    FX StatArb Collector (EUR/GBP census, ended 2026-09-11)"
}
if (-not $DryRun) {
    Get-ScheduledTask -TaskName 'FX StatArb*' | Select-Object TaskName, State | Format-Table -AutoSize
}
