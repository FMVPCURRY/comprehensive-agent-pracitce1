$ErrorActionPreference = "SilentlyContinue"

$project = "D:\third2\comprehensive\Baseline\ChiFraud-main"
$codeProcesses = Get-Process Code | Where-Object { $_.MainWindowHandle -ne 0 } | Sort-Object StartTime -Descending

if (-not $codeProcesses) {
    Start-Process -FilePath "code" -ArgumentList "`"$project`""
    exit 0
}

$signature = @"
using System;
using System.Runtime.InteropServices;

public static class WindowTools {
    [DllImport("user32.dll")]
    public static extern bool SetForegroundWindow(IntPtr hWnd);

    [DllImport("user32.dll")]
    public static extern bool ShowWindowAsync(IntPtr hWnd, int nCmdShow);
}
"@

Add-Type -TypeDefinition $signature -ErrorAction SilentlyContinue

$window = $codeProcesses[0].MainWindowHandle
[WindowTools]::ShowWindowAsync($window, 9) | Out-Null
[WindowTools]::SetForegroundWindow($window) | Out-Null
