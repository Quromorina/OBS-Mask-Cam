<#
.SYNOPSIS
  反応しなくなった USB カメラを、PC を再起動せず・ケーブルを抜き挿しせずにリセットする。

.DESCRIPTION
  USB カメラ（例: Logicool C920）を「抜いて挿し直した」のと同じ状態に戻し、
  Windows のカメラ共有サービス（Frame Server）も再起動する。管理者権限が必要なので、
  権限がなければ自分を管理者として起動し直す（確認ダイアログが 1 回出る）。

.PARAMETER Name
  対象カメラの名前（ワイルドカード可）。省略時は、つながっている USB カメラすべて。

.PARAMETER List
  対象になるカメラを表示するだけで、リセットはしない（管理者権限は不要）。

.EXAMPLE
  .\reset-camera.ps1
  .\reset-camera.ps1 -Name '*C920*'
  .\reset-camera.ps1 -List
#>
param(
    [string]$Name = '*',
    [switch]$List
)

function Get-CameraTargets([string]$Pattern) {
    # 仮想カメラ（OBS Virtual Camera など）は USB ではないので対象外
    $cameras = Get-PnpDevice -Class Camera, Image -PresentOnly -ErrorAction SilentlyContinue |
        Where-Object { $_.InstanceId -like 'USB\*' -and $_.FriendlyName -like $Pattern }
    $targets = @{}
    foreach ($cam in $cameras) {
        $id = $cam.InstanceId
        if ($id -match '&MI_\d+') {
            # 複合デバイス（カメラ＋マイク）の一部なら、USB デバイス全体をリセットする
            $parent = (Get-PnpDeviceProperty -InstanceId $id -KeyName DEVPKEY_Device_Parent -ErrorAction SilentlyContinue).Data
            if ($parent) { $id = $parent }
        }
        $targets[$id] = $cam.FriendlyName
    }
    return $targets
}

$targets = Get-CameraTargets $Name
if ($targets.Count -eq 0) {
    Write-Host "対象の USB カメラが見つかりません（名前: $Name）。カメラがつながっているか確認してください。"
    if (-not $List) { Read-Host 'Enter で閉じます' | Out-Null }
    exit 1
}

if ($List) {
    foreach ($id in $targets.Keys) { Write-Host "$($targets[$id])  [$id]" }
    exit 0
}

$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host '管理者権限で起動し直します...'
    $argList = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$PSCommandPath`"", '-Name', "`"$Name`"")
    try {
        Start-Process powershell.exe -Verb RunAs -ArgumentList $argList
    } catch {
        Write-Host '管理者権限での起動がキャンセルされました。リセットは行っていません。'
        Read-Host 'Enter で閉じます' | Out-Null
        exit 1
    }
    exit 0
}

$failed = 0
foreach ($id in $targets.Keys) {
    Write-Host "リセット中: $($targets[$id])"
    & pnputil.exe /restart-device "$id" | Out-Host
    if ($LASTEXITCODE -ne 0) {
        $failed++
        Write-Host "  失敗しました（終了コード $LASTEXITCODE）。カメラを抜いて挿し直してください。"
    }
}

# Windows の設定画面やカメラアプリが使う Frame Server も、古い状態を持ったままのことがある
foreach ($svc in 'FrameServerMonitor', 'FrameServer') {
    $s = Get-Service -Name $svc -ErrorAction SilentlyContinue
    if ($s -and $s.Status -eq 'Running') {
        try {
            Restart-Service -Name $svc -Force -ErrorAction Stop -WarningAction SilentlyContinue
            Write-Host "サービスを再起動しました: $svc"
        } catch {
            # カメラを使っているアプリがあると止められない。カメラのリセットは済んでいるので続ける
            Write-Host "サービス $svc は使用中のアプリがあるため再起動しませんでした（カメラのリセットには影響しません）"
        }
    }
}

if ($failed -eq 0) {
    Write-Host ''
    Write-Host '完了しました。OBS Mask Cam を起動中なら、数秒で自動的に映像が戻ります。'
}
Read-Host 'Enter で閉じます' | Out-Null
exit $failed
