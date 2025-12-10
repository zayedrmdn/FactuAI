# Context Generator for FactuAI
# Usage: .\context.ps1 (run from scripts/ directory)

# --- CONFIGURATION ---
# Get project root (parent of scripts folder)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$outputFile = Join-Path $scriptDir "context_output.txt"

# Recent Activity Settings
$MaxFilesToRead = 0
$HoursBack = 1

# Target folders to scan for recent activity
$targetFolders = @("frontend", "backend", ".github")

# Folders to completely hide from tree
$foldersToHide = @(".git", ".next", "node_modules", "__pycache__", "unsloth_compiled_cache", "uploads", ".venv-cloud", ".vscode", ".cache")

# Folders to collapse (show name only, hide content)
$collapsedFolders = @("node_modules", ".next", ".git", "dist", "build", "__pycache__", "unsloth_compiled_cache", "uploads", "tests", "scripts")

# Core folders to fully expand (show recursive structure)
$coreFolders = @("backend", "frontend")

# Priority files (ALWAYS include full content, regardless of modification date)
$priorityFiles = @(
    "frontend/package.json",
    "frontend/next.config.ts",
    "frontend/tsconfig.json",
    "requirements.txt",
    "backend/app.py",
    "backend/config.py",
    "README.md",
    "SETUP_INSTRUCTIONS.md"
)

# Database schema references (pointer only, no content)
$dbReferences = @()

# --- FUNCTIONS ---

function Get-SmartTree {
    param ($Path, $Indent = "", $IsCore = $false, $IsRoot = $false)
    $str = ""
    $items = Get-ChildItem -Path $Path -Directory -ErrorAction SilentlyContinue | 
        Where-Object { $foldersToHide -notcontains $_.Name }
    $files = Get-ChildItem -Path $Path -File -ErrorAction SilentlyContinue

    if ($IsCore -or $IsRoot) {
        foreach ($file in $files) {
            $str += "$Indent|-- $($file.Name)`n"
        }
    }
    foreach ($folder in $items) {
        if ($collapsedFolders -contains $folder.Name) {
            $str += "$Indent|-- [$($folder.Name)/] (Content Hidden)`n"
        } elseif ($IsRoot -and $coreFolders -contains $folder.Name) {
            $str += "$Indent|-- [$($folder.Name)/]`n"
            $str += Get-SmartTree -Path $folder.FullName -Indent "$Indent    " -IsCore $true
        } elseif ($IsCore) {
            $str += "$Indent|-- [$($folder.Name)/]`n"
            $str += Get-SmartTree -Path $folder.FullName -Indent "$Indent    " -IsCore $true
        } else {
            $str += "$Indent|-- [$($folder.Name)/] (Content Hidden)`n"
        }
    }
    return $str
}

# --- EXECUTION ---

Set-Location $projectRoot
Write-Host "`n=== Context Generator Started ===" -ForegroundColor Cyan
Write-Host "Project: FactuAI`n" -ForegroundColor Gray

$output = @()
$output += "=== PROJECT CONTEXT FOR AI ==="
$output += "Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
$output += "Project: FactuAI"
$output += ""

# 1. Project Structure
Write-Host "[1/4] Mapping project structure..." -ForegroundColor Yellow
$output += "--- PROJECT STRUCTURE ---"
$output += Get-SmartTree -Path $projectRoot -IsRoot $true
$output += ""

# 2. Database References
Write-Host "[2/4] Adding database schema references..." -ForegroundColor Yellow
$output += "--- DATABASE SCHEMA FILES (Reference Only) ---"
foreach ($dbRef in $dbReferences) {
    $fullPath = Join-Path $projectRoot $dbRef
    if (Test-Path $fullPath) {
        $output += "Location: $dbRef"
    }
}
$output += ""

# 3. Priority Files (Always include)
Write-Host "[3/4] Reading priority configuration files..." -ForegroundColor Yellow
$output += "--- CRITICAL CONFIGURATION FILES ---"
foreach ($priorityPath in $priorityFiles) {
    $fullPath = Join-Path $projectRoot $priorityPath
    if (Test-Path $fullPath) {
        Write-Host "  + $priorityPath" -ForegroundColor Green
        $output += ""
        $output += "=== FILE: $priorityPath ==="
        $output += Get-Content $fullPath -Raw
        $output += "=== END FILE ==="
        $output += ""
    }
}

# 4. Recently Modified Files
Write-Host "[4/4] Scanning recent activity (last $HoursBack hours)..." -ForegroundColor Yellow
$cutoffDate = (Get-Date).AddHours(-$HoursBack)

$recentFiles = Get-ChildItem -Path $targetFolders -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object {
        $_.LastWriteTime -gt $cutoffDate -and
        $_.Extension -in @('.js', '.ts', '.tsx', '.json', '.css', '.md', '.py') -and
        -not ($priorityFiles -contains ($_.FullName.Replace("$projectRoot\", "").Replace("\", "/")))
    } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First $MaxFilesToRead

if ($recentFiles) {
    $output += "--- RECENTLY MODIFIED FILES (Last $HoursBack hours) ---"
    $output += "Found: $($recentFiles.Count) files"
    $output += ""
    
    foreach ($file in $recentFiles) {
        $relativePath = $file.FullName.Replace("$projectRoot\", "").Replace("\", "/")
        Write-Host "  + $relativePath" -ForegroundColor Green
        $output += ""
        $output += "=== FILE: $relativePath ==="
        $output += "Last Modified: $($file.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))"
        $output += "---"
        $output += Get-Content $file.FullName -Raw
        $output += "=== END FILE ==="
        $output += ""
    }
} else {
    Write-Host "  (No recent activity found)" -ForegroundColor Yellow
    $output += "--- NO RECENT ACTIVITY ---"
    $output += ""
}

# --- SAVE AND COPY ---
$finalOutput = $output -join "`n"
$finalOutput | Out-File $outputFile -Encoding UTF8
Set-Clipboard -Value $finalOutput

Write-Host "`n✅ COMPLETE!" -ForegroundColor Green
Write-Host "Output saved to: $outputFile" -ForegroundColor Gray
Write-Host "Context copied to clipboard - Ready to paste into AI!" -ForegroundColor Cyan
Write-Host ""