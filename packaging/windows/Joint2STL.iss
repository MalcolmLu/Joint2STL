#define ProductName "Joint2STL"
#define ProductVersion "0.4.4"

[Setup]
AppId={{9A35D9A2-4EBE-49B6-8477-635A32AB9DA6}
AppName={#ProductName}
AppVersion={#ProductVersion}
AppPublisher=Joint2STL
AppVerName=Joint2STL 0.4.4
DefaultDirName={localappdata}\Programs\Joint2STL
DefaultGroupName=Joint2STL
PrivilegesRequired=lowest
ArchitecturesAllowed=x64os
ArchitecturesInstallIn64BitMode=x64os
MinVersion=10.0.17763
DisableProgramGroupPage=yes
DisableDirPage=no
UsePreviousAppDir=yes
UsePreviousTasks=yes
SetupIconFile=payload\Joint2STL.ico
UninstallDisplayIcon={app}\Joint2STL.ico
UninstallDisplayName=Joint2STL 0.4.4
WizardStyle=modern
ChangesEnvironment=no
AllowNoIcons=no
CloseApplications=yes
RestartApplications=no
Compression=lzma2/normal
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=6
SolidCompression=yes
VersionInfoVersion=0.4.4.0
VersionInfoDescription=Joint2STL 0.4.4 Windows x64 Offline Setup
VersionInfoProductName=Joint2STL
VersionInfoProductVersion=0.4.4
OutputDir=packages
OutputBaseFilename=Joint2STL-0.4.4-Windows-x64-Setup
#ifdef SplitMedia
DiskSpanning=yes
DiskSliceSize=max
SlicesPerDisk=1
#endif

[Languages]
Name: "zh_CN"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"; InfoBeforeFile: "payload\安装说明.txt"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "快捷方式："; Flags: unchecked

[Files]
Source: "payload\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Joint2STL"; Filename: "{app}\runtime\pythonw.exe"; Parameters: "-I -B -X utf8 ""{app}\start-joint2stl.pyw"""; WorkingDir: "{app}"; IconFilename: "{app}\Joint2STL.ico"; Comment: "打开 Joint2STL 骨模工作台"
Name: "{userdesktop}\Joint2STL"; Filename: "{app}\runtime\pythonw.exe"; Parameters: "-I -B -X utf8 ""{app}\start-joint2stl.pyw"""; WorkingDir: "{app}"; IconFilename: "{app}\Joint2STL.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\runtime\pythonw.exe"; Parameters: "-I -B -X utf8 ""{app}\start-joint2stl.pyw"""; WorkingDir: "{app}"; Description: "启动 Joint2STL"; Flags: nowait postinstall skipifsilent unchecked
