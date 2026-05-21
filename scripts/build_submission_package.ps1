param(
    [string]$PackageName = 'XXX-University-TeamLeader-Phone'
)

$ErrorActionPreference = 'Stop'

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Dist = Join-Path $Root 'dist'
$PackageDir = Join-Path $Dist $PackageName
$SourceDir = Join-Path $PackageDir 'source_code'
$ResultsDir = Join-Path $PackageDir 'results'

if (Test-Path -LiteralPath $PackageDir) {
    Remove-Item -LiteralPath $PackageDir -Recurse -Force
}

New-Item -ItemType Directory -Force -Path $SourceDir | Out-Null
New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null

$topLevelFiles = @(
    'README.md',
    'requirements.txt',
    'run_demo.bat',
    'stop_demo.bat',
    'SUBMISSION_CHECKLIST.md'
)

foreach ($relative in $topLevelFiles) {
    $source = Join-Path $Root $relative
    if (Test-Path -LiteralPath $source) {
        Copy-Item -LiteralPath $source -Destination $PackageDir -Force
        Copy-Item -LiteralPath $source -Destination $SourceDir -Force
    }
}

$sourceItems = @(
    'config',
    'docs',
    'scripts',
    'src',
    'web',
    'tests'
)

foreach ($relative in $sourceItems) {
    $source = Join-Path $Root $relative
    if (Test-Path -LiteralPath $source) {
        Copy-Item -LiteralPath $source -Destination $SourceDir -Recurse -Force
    }
}

$resultPatterns = @(
    'demo_evaluation.json',
    'public_ntu_evaluation.json',
    'submission_readiness.json',
    '*.md'
)

foreach ($pattern in $resultPatterns) {
    Get-ChildItem -LiteralPath (Join-Path $Root 'results') -Filter $pattern -File -ErrorAction SilentlyContinue |
        ForEach-Object {
            Copy-Item -LiteralPath $_.FullName -Destination $ResultsDir -Force
        }
}

$packageReadme = @(
    '# Submission Package',
    '',
    'This package is generated for the XH-202617 competition submission workflow.',
    '',
    'Items still requiring real team information before final submission:',
    '',
    '- school, team leader, phone number, members, advisors',
    '- approved registration form scan',
    '- EZVIZ Open Platform call records or device test evidence',
    '- real video or public dataset evaluation results',
    '',
    'Run the demo:',
    '',
    '```powershell',
    'cd source_code',
    '.\run_demo.bat',
    '```',
    '',
    'Run the built-in evaluation:',
    '',
    '```powershell',
    'cd source_code',
    'python -B scripts\evaluate_competition_demo.py --repeats 5',
    '```'
) -join "`r`n"

Set-Content -LiteralPath (Join-Path $PackageDir 'PACKAGE_README.md') -Value $packageReadme -Encoding ASCII

Write-Host 'Submission package generated:'
Write-Host $PackageDir
