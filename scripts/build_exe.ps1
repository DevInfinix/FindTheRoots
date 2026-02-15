param(
    [switch]$UseHiddenImports
)

$python = ".\.venv\Scripts\python"
if (-not (Test-Path $python)) {
    throw "Virtual environment not found at .venv"
}

$command = @(
    "$python",
    "-m",
    "PyInstaller",
    "--onefile",
    "--windowed"
)

if ($UseHiddenImports) {
    $command += "--hidden-import"
    $command += "sympy"
    $command += "--hidden-import"
    $command += "matplotlib"
}

$command += "main.py"

Write-Host "Running: $($command -join ' ')"
& $command[0] $command[1..($command.Length - 1)]
