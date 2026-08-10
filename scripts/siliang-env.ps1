[CmdletBinding(DefaultParameterSetName = 'Show')]
param(
    [Parameter(Mandatory = $true, ParameterSetName = 'Enable')]
    [ValidateRange(1, [int]::MaxValue)]
    [int] $CacheMiB,

    [Parameter(Mandatory = $true, ParameterSetName = 'Disable')]
    [switch] $Disable,

    [Parameter(Mandatory = $true, ParameterSetName = 'Reset')]
    [switch] $Reset,

    [Parameter(ParameterSetName = 'Show')]
    [switch] $Show,

    [Parameter(ParameterSetName = 'Enable')]
    [switch] $NoMemoryReport
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$managedNames = @(
    'SILIANGEM_CACHE_MIB',
    'SILIANGEM_DEFER',
    'SILIANGEM_DISABLE',
    'SILIANGEM_MEM_REPORT',
    'SILIANGEM_SLAB',
    'SILIANGEM_VERBOSE',
    'GGML_MOE_PREFETCH'
)

function Clear-SiliangEnvironment {
    foreach ($name in $managedNames) {
        Remove-Item -LiteralPath ("Env:{0}" -f $name) -ErrorAction SilentlyContinue
    }
}

function Show-SiliangEnvironment {
    foreach ($name in $managedNames) {
        $value = [Environment]::GetEnvironmentVariable($name, 'Process')
        [pscustomobject]@{
            Name  = $name
            Value = if ($null -eq $value) { '<unset>' } else { $value }
        }
    }
}

switch ($PSCmdlet.ParameterSetName) {
    'Enable' {
        Clear-SiliangEnvironment
        $env:SILIANGEM_CACHE_MIB = $CacheMiB.ToString([Globalization.CultureInfo]::InvariantCulture)
        $env:SILIANGEM_DEFER = '1'
        $env:GGML_MOE_PREFETCH = '0'

        if ($PSBoundParameters.ContainsKey('Verbose') -and [bool] $PSBoundParameters['Verbose']) {
            $env:SILIANGEM_VERBOSE = '1'
        }
        if ($NoMemoryReport) {
            $env:SILIANGEM_MEM_REPORT = '0'
        }

        Write-Output 'Siliang environment: arena enabled for this PowerShell process.'
    }
    'Disable' {
        Clear-SiliangEnvironment
        $env:SILIANGEM_DISABLE = '1'
        $env:GGML_MOE_PREFETCH = '0'
        Write-Output 'Siliang environment: arena disabled; mmap control selected.'
    }
    'Reset' {
        Clear-SiliangEnvironment
        Write-Output 'Siliang environment: reset.'
    }
    'Show' {
        Write-Output 'Siliang environment: current process values.'
    }
    default {
        throw "Unexpected parameter set: $($PSCmdlet.ParameterSetName)"
    }
}

Show-SiliangEnvironment
