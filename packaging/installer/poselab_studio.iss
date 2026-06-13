; PoseLab Studio オンラインインストーラー (Inno Setup スクリプト)
;
; 小さな Setup.exe を作る。インストール時に setup_env.ps1 が走り、
; uv で専用 Python + PyTorch (GPU 自動判定) + mmpose + poselab を
; ユーザーのマシンに構築する。Python / pip の事前インストールは不要。
;
; ビルド: ISCC.exe poselab_studio.iss /DAppVersion=0.6.1
; (通常は .github/workflows/build-installer.yml が CI で実行する)

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{6F9A1C2E-3B7D-4E55-9C1A-POSELABSTUDIO}}
AppName=PoseLab Studio
AppVersion={#AppVersion}
AppPublisher=cfn0eft
; インストール先はスペースを含めない (uv.exe が空白入りパス引数を誤解釈するため)。
; 表示名・ショートカット名は "PoseLab Studio" のまま。
DefaultDirName={autopf}\PoseLabStudio
DefaultGroupName=PoseLab Studio
DisableProgramGroupPage=yes
DisableDirPage=no
; 管理者権限なしでユーザー領域に入れる (pip 書き込みでつまずかない)
PrivilegesRequired=lowest
OutputDir=Output
OutputBaseFilename=PoseLabStudioSetup
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
; インストール時に約0.6〜2GB を取得するため十分な空きを要求
ExtraDiskSpaceRequired=6442450944

[Languages]
Name: "japanese"; MessagesFile: "compiler:Languages\Japanese.isl"

[Tasks]
Name: "desktopicon"; Description: "デスクトップにショートカットを作成する"; GroupDescription: "追加タスク:"

[Files]
Source: "uv.exe";              DestDir: "{app}"; Flags: ignoreversion
Source: "setup_env.ps1";       DestDir: "{app}"; Flags: ignoreversion
Source: "PoseLabStudio.cmd";   DestDir: "{app}"; Flags: ignoreversion
Source: "wheels\*";            DestDir: "{app}\wheels"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\PoseLab Studio"; Filename: "{app}\PoseLabStudio.cmd"; WorkingDir: "{app}"; Comment: "PoseLab Studio を起動"
Name: "{group}\PoseLab Studio をアンインストール"; Filename: "{uninstallexe}"
Name: "{autodesktop}\PoseLab Studio"; Filename: "{app}\PoseLabStudio.cmd"; WorkingDir: "{app}"; Tasks: desktopicon

[UninstallDelete]
Type: filesandordirs; Name: "{app}\env"
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\uv-cache"
Type: filesandordirs; Name: "{app}\wheels"
Type: files;          Name: "{app}\constraints.txt"

[Code]
function InitializeSetup(): Boolean;
begin
  if MsgBox('インストール時に必要なコンポーネント (Python・PyTorch・mmpose など) を'
    + #13#10 + 'インターネットから取得します (GPU 搭載機で約2GB、非搭載機で約0.6GB)。'
    + #13#10#13#10 + 'インターネットに接続した状態で続行してください。続けますか?',
    mbConfirmation, MB_YESNO) = IDYES then
    Result := True
  else
    Result := False;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
  Params: String;
begin
  if CurStep = ssPostInstall then
  begin
    Params := '-NoProfile -ExecutionPolicy Bypass -File "'
      + ExpandConstant('{app}\setup_env.ps1') + '" -InstallDir "'
      + ExpandConstant('{app}') + '"';
    if not Exec('powershell.exe', Params, '', SW_SHOW, ewWaitUntilTerminated, ResultCode) then
    begin
      MsgBox('セットアップスクリプトを起動できませんでした。', mbCriticalError, MB_OK);
      Abort;
    end;
    if ResultCode <> 0 then
    begin
      MsgBox('コンポーネントのセットアップに失敗しました (コード ' + IntToStr(ResultCode) + ')。'
        + #13#10 + 'インターネット接続を確認してから、もう一度インストールしてください。',
        mbCriticalError, MB_OK);
      Abort;
    end;
  end;
end;
