Option Explicit
'
' Launch the batch file named in the first argument with NO visible console
' window. Same mechanism as launch_collector.vbs (see its header for why), but
' parameterised so each Phase 1b arm reuses it:
'
'     wscript.exe launch_arm.vbs run_collector_audnzd.bat
'
' Waits for the batch and propagates its exit code, so the task stays Running
' for the life of the collector and restart-on-failure sees real failures.
'
Dim fso, sh, here, rc
If WScript.Arguments.Count < 1 Then WScript.Quit 2
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
here = fso.GetParentFolderName(WScript.ScriptFullName)
rc = sh.Run("""" & here & "\" & WScript.Arguments(0) & """", 0, True)
WScript.Quit rc
