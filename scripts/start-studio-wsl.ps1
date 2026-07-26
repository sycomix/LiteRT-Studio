param(
    [string]$Distro = "Ubuntu-24.04",
    [int]$Port = 7860
)

$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$wslWorkspace = wsl -d $Distro -u root wslpath -a $workspace
if ($LASTEXITCODE -ne 0) {
    throw "Could not resolve the project path inside $Distro."
}

wsl -d $Distro -u root bash -lc "cd '$wslWorkspace' && /root/litert-studio-venv/bin/python -m pip install -e '.[api,training,conversion,runtime]' && /root/litert-studio-venv/bin/litert-studio serve --workspace . --port $Port"
