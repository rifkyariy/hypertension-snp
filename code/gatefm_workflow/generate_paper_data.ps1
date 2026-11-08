param(
    [string]$Python = "python",
    [string]$Dimensions = "20,100,500,1000,5000,10000",
    [string]$DataSeeds = "42,100,200,300,400",
    [string]$GametesJar = "",
    [switch]$SkipExisting,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$PackageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ExpectedSha256 = "5E38B0993F8915E3E3E4C225CE6B684455D02C6189877D6AE219D24AD69E27D7"
$DownloadUrl = "https://sourceforge.net/projects/gametes/files/GAMETES_2.1.jar/download"

if (-not (Get-Command java -ErrorAction SilentlyContinue)) {
    throw "Java was not found on PATH. Install a Java runtime before generating GAMETES data."
}

if ([string]::IsNullOrWhiteSpace($GametesJar)) {
    $ToolsDir = Join-Path $PackageRoot "tools"
    New-Item -ItemType Directory -Path $ToolsDir -Force | Out-Null
    $ResolvedJar = Join-Path $ToolsDir "GAMETES_2.1.jar"
    if (-not (Test-Path -LiteralPath $ResolvedJar)) {
        $DownloadPath = Join-Path $ToolsDir "GAMETES_2.1.jar.download"
        Write-Host "Downloading GAMETES 2.1 from the official SourceForge project..."
        Invoke-WebRequest -Uri $DownloadUrl -OutFile $DownloadPath -UseBasicParsing
        $DownloadedHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $DownloadPath).Hash
        if ($DownloadedHash -ne $ExpectedSha256) {
            $file = [System.IO.File]::OpenRead($DownloadPath)
            try {
                $header = [byte[]]::new(16)
                $length = $file.Read($header, 0, $header.Length)
            } finally {
                $file.Dispose()
            }
            $isZip = $length -ge 4 -and $header[0] -eq 0x50 -and $header[1] -eq 0x4B
            Remove-Item -LiteralPath $DownloadPath -Force
            if (-not $isZip) {
                throw ("SourceForge returned a download webpage rather than a JAR. " +
                       "Open $DownloadUrl in your browser, save GAMETES_2.1.jar, " +
                       "then rerun with -GametesJar <path-to-jar>. " +
                       "Expected SHA-256: $ExpectedSha256")
            }
            throw "GAMETES jar checksum mismatch. Expected SHA-256: $ExpectedSha256; downloaded: $DownloadedHash"
        }
        Move-Item -LiteralPath $DownloadPath -Destination $ResolvedJar
    }
} else {
    $ResolvedJar = (Resolve-Path -LiteralPath $GametesJar).Path
}

$JarHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $ResolvedJar).Hash
if ($JarHash -ne $ExpectedSha256) {
    throw "GAMETES jar checksum mismatch. Expected $ExpectedSha256, found $JarHash."
}

$Arguments = @(
    (Join-Path $PackageRoot "code\generate_gametes_data.py"),
    "--gametes-jar", $ResolvedJar,
    "--output-root", (Join-Path $PackageRoot "data"),
    "--dimensions", $Dimensions,
    "--data-seeds", $DataSeeds
)
if ($SkipExisting) {
    $Arguments += "--skip-existing"
}
if ($DryRun) {
    $Arguments += "--dry-run"
}

& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "GAMETES data generation failed with exit code $LASTEXITCODE."
}
