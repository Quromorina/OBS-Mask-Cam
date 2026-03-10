; OBS Mask Cam Inno Setup Script

[Setup]
; ユニークなID（アンインストール登録用。これによって「プログラムと機能」から綺麗に削除可能になる）
AppId={{867FE60D-D57B-4929-B583-05562BB7AEDC}}
; アプリケーション情報
AppName=OBS Mask Cam
AppVersion=1.0.0
AppPublisher=OBS Mask Cam Project
; デフォルトのインストール先 (Program Files配下)
DefaultDirName={autopf}\OBSMaskCam
; スタートメニューのフォルダ名
DefaultGroupName=OBS Mask Cam
; インストーラーの出力先ディレクトリ
OutputDir=.\InnoSetup_Output
; 生成されるインストーラーのファイル名
OutputBaseFilename=OBSMaskCam_Setup_v1.0.0
; 圧縮方法
Compression=lzma2
SolidCompression=yes
; 特権の要求（Program Filesにインストールするため管理者権限が必要）
PrivilegesRequired=admin
; アンインストーラー対応
UninstallDisplayIcon={app}\OBSMaskCam.exe
; アイコン設定（任意）
SetupIconFile=icon.ico

[Tasks]
; デスクトップアイコンを作成するかどうかのチェックボックス
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; EXEファイルと_internalフォルダの中身をすべてコピー
Source: "dist\OBSMaskCam\OBSMaskCam.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist\OBSMaskCam\_internal\*"; DestDir: "{app}\_internal"; Flags: ignoreversion recursesubdirs createallsubdirs
; マスクフォルダもコピー
Source: "masks\*"; DestDir: "{app}\masks"; Flags: ignoreversion recursesubdirs createallsubdirs
; アイコン
Source: "icon.ico"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
; スタートメニューのショートカット
Name: "{group}\OBS Mask Cam"; Filename: "{app}\OBSMaskCam.exe"; IconFilename: "{app}\icon.ico"
Name: "{group}\{cm:UninstallProgram,OBS Mask Cam}"; Filename: "{uninstallexe}"
; デスクトップショートカット（Taskの選択次第）
Name: "{autodesktop}\OBS Mask Cam"; Filename: "{app}\OBSMaskCam.exe"; Tasks: desktopicon; IconFilename: "{app}\icon.ico"

[Run]
; インストール完了後に起動するオプション
Filename: "{app}\OBSMaskCam.exe"; Description: "{cm:LaunchProgram,OBS Mask Cam}"; Flags: nowait postinstall skipifsilent
