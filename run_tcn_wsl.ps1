param(
    [string]$Distro = "Ubuntu"
)

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$resolvedRepoRoot = (Resolve-Path $repoRoot).Path

$driveLetter = $resolvedRepoRoot.Substring(0, 1).ToLowerInvariant()
$wslPathTail = $resolvedRepoRoot.Substring(2).Replace('\', '/')
$wslRepoRoot = "/mnt/$driveLetter$wslPathTail"

$bashCommand = "cd '$wslRepoRoot' && bash ./run_tcn_wsl.sh"
wsl -d $Distro -- bash -lc $bashCommand
