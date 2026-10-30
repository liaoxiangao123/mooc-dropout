# Update the TOC field in the thesis .docx and export a PDF via Word automation.
#
# Why: the table of contents in the docx is a TOC field; python-docx can only
# write the field code, the field *result* has to be computed by Word. Exporting
# without updating leaves the placeholder line
#   "[TOC: right-click in Word -> Update Field]"
# in the PDF, which is exactly what happened to the previous export.
#
# Usage (from project root):
#   powershell -ExecutionPolicy Bypass -File scripts/export_pdf.ps1
# The script picks the only .docx under docs/ unless -Docx is given.
#
# NOTE: keep this file ASCII-only. Windows PowerShell 5.1 reads .ps1 as ANSI
# when there is no BOM, so any Chinese literal here would be garbled.

param(
    [string]$Docx = "",
    [string]$Pdf  = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$docsDir = Join-Path $root "docs"

if ($Docx -eq "") {
    $cand = Get-ChildItem -Path $docsDir -Filter "*.docx" | Where-Object { $_.Name -notlike "~*" }
    if ($cand.Count -eq 0) { throw "no .docx found under docs/" }
    $Docx = $cand[0].FullName
}
if ($Pdf -eq "") {
    $Pdf = [System.IO.Path]::ChangeExtension($Docx, ".pdf")
}

Write-Output ("docx: " + $Docx)
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($Docx, $false, $false)
    if ($null -eq $doc) { throw "Word failed to open the document" }
    $doc.Fields.Update() | Out-Null
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() | Out-Null }
    $doc.Repaginate()
    Write-Output ("sections: " + $doc.Sections.Count + " / toc fields: " + $doc.TablesOfContents.Count)
    $doc.Save()
    # 参数依次：输出文件、PDF、导出后不打开、按打印优化、全文、页码范围、导出正文、
    # 含文档属性、保留 IRM、按标题建书签（便于答辩时跳页）、结构标签、位图缺失字体、不转 PDF/A
    $doc.ExportAsFixedFormat($Pdf, 17, $false, 0, 0, 1, 1, 0, $true, $true, 1, $true, $true, $false)
    $doc.Close($false)
    Write-Output ("pdf: " + $Pdf)
}
finally {
    $word.Quit()
    [void][Runtime.InteropServices.Marshal]::ReleaseComObject($word)
}
