# Image Corruption Analyzer
# Tests actual image readability, not just file headers
# Joel Morin - VE1ATM - 2025

param(
    [string]$Path = "D:\broken",
    [string]$OutputPath = "D:\broken\validation_report.txt"
)

Write-Host "Image Corruption Analyzer" -ForegroundColor Cyan
Write-Host "=========================" -ForegroundColor Cyan
Write-Host "Scanning: $Path" -ForegroundColor Yellow
Write-Host ""

# Results tracking
$results = @{
    Good = @()
    HeaderOnly = @()
    Truncated = @()
    Corrupted = @()
    TotalSize = 0
}

# Get all image files
$imageFiles = Get-ChildItem -Path $Path -File | Where-Object {
    $_.Extension -match '\.(jpg|jpeg|tif|tiff|png|bmp|gif)$'
}

$total = $imageFiles.Count
$current = 0

foreach ($file in $imageFiles) {
    $current++
    $percent = [math]::Round(($current / $total) * 100)
    Write-Progress -Activity "Validating Images" -Status "$current of $total" -PercentComplete $percent
    
    $result = @{
        Name = $file.Name
        SizeMB = [math]::Round($file.Length / 1MB, 2)
        Status = "Unknown"
        Details = ""
    }
    
    $results.TotalSize += $file.Length
    
    # Read file header to check magic bytes
    try {
        $bytes = [System.IO.File]::ReadAllBytes($file.FullName)
        
        # Check for valid image headers
        $hasValidHeader = $false
        $headerType = ""
        
        # JPEG: FF D8
        if ($bytes[0] -eq 0xFF -and $bytes[1] -eq 0xD8) {
            $hasValidHeader = $true
            $headerType = "JPEG"
        }
        # TIFF: II (little-endian) or MM (big-endian)
        elseif (($bytes[0] -eq 0x49 -and $bytes[1] -eq 0x49) -or 
                ($bytes[0] -eq 0x4D -and $bytes[1] -eq 0x4D)) {
            $hasValidHeader = $true
            $headerType = "TIFF"
        }
        # PNG: 89 50 4E 47
        elseif ($bytes[0] -eq 0x89 -and $bytes[1] -eq 0x50 -and 
                $bytes[2] -eq 0x4E -and $bytes[3] -eq 0x47) {
            $hasValidHeader = $true
            $headerType = "PNG"
        }
        
        if (-not $hasValidHeader) {
            $result.Status = "Corrupted"
            $result.Details = "Invalid file header - not a recognized image format"
            $results.Corrupted += $result
            continue
        }
        
        # Check if file is mostly zeros (common in failed recovery)
        $zeroCount = ($bytes | Where-Object { $_ -eq 0 }).Count
        $zeroPercent = [math]::Round(($zeroCount / $bytes.Length) * 100, 1)
        
        if ($zeroPercent -gt 50) {
            $result.Status = "Corrupted"
            $result.Details = "File is $zeroPercent% null bytes - likely unrecoverable"
            $results.Corrupted += $result
            continue
        }
        
        # Try to load with .NET imaging
        try {
            $img = [System.Drawing.Image]::FromFile($file.FullName)
            
            # Successfully loaded - this is a GOOD file
            $result.Status = "Good"
            $result.Details = "✅ Fully readable - $($img.Width)x$($img.Height) $headerType"
            $results.Good += $result
            $img.Dispose()
        }
        catch {
            $errorMsg = $_.Exception.Message
            
            # Try to determine type of corruption
            if ($errorMsg -match "truncated|incomplete|premature|end of") {
                $result.Status = "Truncated"
                $result.Details = "⚠️  Valid header, but image data is incomplete/truncated"
                $results.Truncated += $result
            }
            elseif ($file.Length -lt 10KB) {
                $result.Status = "HeaderOnly"
                $result.Details = "⚠️  File too small - likely thumbnail/header only ($($ file.Length) bytes)"
                $results.HeaderOnly += $result
            }
            else {
                $result.Status = "Corrupted"
                $result.Details = "❌ $headerType header present but internal structure corrupted"
                $results.Corrupted += $result
            }
        }
    }
    catch {
        $result.Status = "Corrupted"
        $result.Details = "❌ Cannot read file: $($_.Exception.Message)"
        $results.Corrupted += $result
    }
}

Write-Progress -Activity "Validating Images" -Completed

# Generate Report
$report = @"
IMAGE CORRUPTION ANALYSIS REPORT
Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
Directory: $Path
================================================================================

SUMMARY
-------
Total Files Scanned:    $total
Total Size:             $([math]::Round($results.TotalSize / 1GB, 2)) GB

✅ GOOD (Fully Readable):          $($results.Good.Count) files ($([math]::Round($results.Good.Count / $total * 100, 1))%)
⚠️  TRUNCATED (Partial Data):      $($results.Truncated.Count) files ($([math]::Round($results.Truncated.Count / $total * 100, 1))%)
⚠️  HEADER ONLY (Too Small):       $($results.HeaderOnly.Count) files ($([math]::Round($results.HeaderOnly.Count / $total * 100, 1))%)
❌ CORRUPTED (Unreadable):         $($results.Corrupted.Count) files ($([math]::Round($results.Corrupted.Count / $total * 100, 1))%)

================================================================================
DETAILED RESULTS
================================================================================

"@

# Good Files
if ($results.Good.Count -gt 0) {
    $report += "`n✅ GOOD FILES (Fully Readable - Worth Keeping)`n"
    $report += "-" * 80 + "`n"
    foreach ($item in $results.Good | Sort-Object Name) {
        $report += "{0,-60} {1,8} MB  {2}`n" -f $item.Name, $item.SizeMB, $item.Details
    }
}

# Truncated Files
if ($results.Truncated.Count -gt 0) {
    $report += "`n⚠️  TRUNCATED FILES (Partial Image Data)`n"
    $report += "-" * 80 + "`n"
    $report += "These have valid headers but incomplete image data. Might be partially viewable.`n`n"
    foreach ($item in $results.Truncated | Sort-Object Name) {
        $report += "{0,-60} {1,8} MB  {2}`n" -f $item.Name, $item.SizeMB, $item.Details
    }
}

# Header Only Files
if ($results.HeaderOnly.Count -gt 0) {
    $report += "`n⚠️  HEADER/THUMBNAIL ONLY FILES (Likely Unrecoverable)`n"
    $report += "-" * 80 + "`n"
    $report += "These are very small - likely just metadata/thumbnails without main image data.`n`n"
    foreach ($item in $results.HeaderOnly | Sort-Object Name) {
        $report += "{0,-60} {1,8} MB  {2}`n" -f $item.Name, $item.SizeMB, $item.Details
    }
}

# Corrupted Files
if ($results.Corrupted.Count -gt 0) {
    $report += "`n❌ CORRUPTED FILES (Unrecoverable)`n"
    $report += "-" * 80 + "`n"
    $report += "These have severe corruption and are likely not recoverable.`n`n"
    foreach ($item in $results.Corrupted | Sort-Object Name) {
        $report += "{0,-60} {1,8} MB  {2}`n" -f $item.Name, $item.SizeMB, $item.Details
    }
}

$report += "`n" + "=" * 80 + "`n"
$report += "RECOMMENDATIONS:`n"
$report += "  - KEEP the $($results.Good.Count) GOOD files - they are fully recoverable`n"
if ($results.Truncated.Count -gt 0) {
    $report += "  - REVIEW the $($results.Truncated.Count) TRUNCATED files - some may have partial images`n"
}
$report += "  - DELETE the $($results.HeaderOnly.Count + $results.Corrupted.Count) corrupted files to save space`n"
$report += "=" * 80 + "`n"

# Display report
Write-Host "`n"
Write-Host $report

# Save report
$report | Out-File -FilePath $OutputPath -Encoding UTF8
Write-Host "Report saved to: $OutputPath" -ForegroundColor Green

# Create lists for easy file management
$goodListPath = Join-Path (Split-Path $OutputPath) "good_files.txt"
$badListPath = Join-Path (Split-Path $OutputPath) "bad_files.txt"

$results.Good | ForEach-Object { $_.Name } | Out-File -FilePath $goodListPath -Encoding UTF8
($results.Corrupted + $results.HeaderOnly) | ForEach-Object { $_.Name } | Out-File -FilePath $badListPath -Encoding UTF8

Write-Host "Good files list: $goodListPath" -ForegroundColor Green
Write-Host "Bad files list: $badListPath" -ForegroundColor Green
