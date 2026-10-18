param([Parameter(Mandatory=$true)][string]$InputDocx, [Parameter(Mandatory=$true)][string]$OutputPdf)
$ErrorActionPreference = "Stop"
$docPath = [System.IO.Path]::GetFullPath($InputDocx)
$pdfPath = [System.IO.Path]::GetFullPath($OutputPdf)
$wordApp = $null
$wordDoc = $null
try {
    $wordApp = New-Object -ComObject Word.Application
    $wordApp.Visible = $false
    $wordApp.DisplayAlerts = 0
    $wordDoc = $wordApp.Documents.Open($docPath, $false, $true)
    $wordDoc.Repaginate()
    $wordDoc.ExportAsFixedFormat($pdfPath, 17)
    Write-Output ("Pages: " + $wordDoc.ComputeStatistics(2))
} finally {
    if ($null -ne $wordDoc) { $wordDoc.Close(0) }
    if ($null -ne $wordApp) { $wordApp.Quit() }
}
