Option Explicit
'
' Launch run_collector.bat with NO visible console window.
'
' Why this exists: the scheduled task runs a .bat, which needs cmd.exe, which
' gets a console. Under an Interactive logon that console is a real window on
' the desktop -- and on 2026-09-01 closing it killed the collector and cost 3.4
' hours of the census.
'
' wscript.exe is a GUI-subsystem host with no console of its own, so launching
' the batch from here with intWindowStyle = 0 hides the window entirely.
'
' bWaitOnReturn = True matters for two reasons: the task stays in the "Running"
' state for the life of the collector (so Task Scheduler will not start a second
' one), and the batch's exit code is propagated below, so restart-on-failure
' still sees a real failure.
'
' The cleaner fix is to switch the task's principal to S4U, which runs it in
' session 0 with no desktop at all AND removes the need to stay logged in. That
' requires an elevated prompt; this wrapper does not, and the two compose
' harmlessly if S4U is applied later.
'
Dim fso, sh, here, rc
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
here = fso.GetParentFolderName(WScript.ScriptFullName)
rc = sh.Run("""" & here & "\run_collector.bat""", 0, True)
WScript.Quit rc
