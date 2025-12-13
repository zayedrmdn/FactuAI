# Context Generator for FactuAI
# Usage: .\context.ps1

# --- CONFIGURATION ---
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptPath
$outputFile = Join-Path $scriptPath "context_output.txt"

# Recent Activity Settings
$MaxFilesToRead = 1
$HoursBack = 1

# Target folders to scan for recent activity
# Added specific feature folders to ensure we catch logic changes
$targetFolders = @(
    "backend/app",
    "backend/migrations",
    "frontend/src",
    "docs"
)

# Folders to completely hide from tree (Noise reduction)
$foldersToHide = @(
    ".agent-memory",
    ".trae", 
    ".venv-cloud",
    ".vscode",
    ".github",
    ".idea",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".next",
    "dist",
    "build",
    ".turbo",
    "venv",
    ".venv",
    ".cache",
    "coverage",
    ".husky"
)

# Folders to collapse (Show name only, hide content)
# Updated based on your actual project structure
$collapsedFolders = @(
    "unsloth_compiled_cache",
    "instance",   # backend/instance
    "uploads",    # backend/uploads
    ".git",
    "public",     # frontend/public
    "tests",      # backend/tests
    "scripts",    # backend/scripts
    "migrations"  # backend/migrations (often too noisy for general context)
)

# Priority files (ALWAYS include full content, regardless of modification date)
# Updated paths to match your actual Vertical Slice Architecture
$priorityFiles = @(
    "CONSTITUTION.md",
    "AGENTS.md",
    "backend/requirements-core.txt",
    "backend/app/core/settings.py"  # Crucial for env var config
)

# Database schema references (pointer only, no content)
# Pointing to the actual persistence models found in your tree
$dbReferences = @(
    "backend/app/persistence/models.py",
    "backend/app/features/auth/models.py"
)

# --- FUNCTIONS ---

function Get-SmartTree {
    param ($Path, $Indent = "")
    $str = ""
    # Get directories
    $items = Get-ChildItem -Path $Path -Directory -ErrorAction SilentlyContinue | 
        Where-Object { $foldersToHide -notcontains $_.Name }
    # Get files
    $files = Get-ChildItem -Path $Path -File -ErrorAction SilentlyContinue

    # Print Files first (to look like file explorer), or folders first? 
    # Standard tree is usually folders then files, or mixed. 
    # Let's do Files then Folders to keep directories distinct.
    
    foreach ($file in $files) {
        $str += "$Indent|-- $($file.Name)`n"
    }
    
    foreach ($folder in $items) {
        if ($collapsedFolders -contains $folder.Name) {
            $str += "$Indent|-- [$($folder.Name)/] (Content Hidden)`n"
        } else {
            $str += "$Indent|-- [$($folder.Name)/]`n"
            $str += Get-SmartTree -Path $folder.FullName -Indent "$Indent    "
        }
    }
    return $str
}

# --- EXECUTION ---

Set-Location $projectRoot
Write-Host "`n=== Context Generator Started ===" -ForegroundColor Cyan
Write-Host "Project: FactuAI" -ForegroundColor Gray
Write-Host "Root: $projectRoot`n" -ForegroundColor Gray

$output = @()
$output += "=== PROJECT CONTEXT FOR AI ==="
$output += "Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
$output += "Project: FactuAI (Fact-Checking AI Application)"
$output += "Repository: https://github.com/zayedrmdn/FactuAI"
$output += ""

# 1. Project Structure
Write-Host "[1/4] Mapping project structure..." -ForegroundColor Yellow
$output += "--- PROJECT STRUCTURE ---"
$output += Get-SmartTree -Path $projectRoot
$output += ""

# 2. Database References
Write-Host "[2/4] Adding database schema references..." -ForegroundColor Yellow
$output += "--- DATABASE SCHEMA FILES (Reference Only) ---"
foreach ($dbRef in $dbReferences) {
    $fullPath = Join-Path $projectRoot $dbRef
    if (Test-Path $fullPath) {
        $output += "File: $dbRef"
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
    } else {
        Write-Host "  ! Missing: $priorityPath" -ForegroundColor Red
    }
}

# 4. Recently Modified Files
Write-Host "[4/4] Scanning recent activity (last $HoursBack hours)..." -ForegroundColor Yellow
$cutoffDate = (Get-Date).AddHours(-$HoursBack)

# Helper to check if file is already in priority list (to avoid duplicates)
function Is-Priority ($filePath) {
    foreach ($p in $priorityFiles) {
        if ($filePath.Replace("\", "/").EndsWith($p)) { return $true }
    }
    return $false
}

$recentFiles = Get-ChildItem -Path $targetFolders -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object {
        $_.LastWriteTime -gt $cutoffDate -and
        $_.Extension -in @('.py', '.tsx', '.ts', '.css', '.js', '.json', '.md', '.sql') -and
        $_.Name -notmatch '\.pyc$|\.map$|\.lock$|package-lock\.json' -and
        -not (Is-Priority $_.FullName)
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