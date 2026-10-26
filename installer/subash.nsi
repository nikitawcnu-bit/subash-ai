; Subash AI — установщик для Windows (NSIS 3.x, Unicode).
; Сборка: powershell -ExecutionPolicy Bypass -File installer\build_setup.ps1

Unicode true
!include "MUI2.nsh"
!include "FileFunc.nsh"

!define APP_NAME "Subash AI"
!define APP_VERSION "0.3.0"
!define APP_PUBLISHER "sunset.ai"
!define APP_URL "https://github.com/nikitawcnu-bit/subash-ai"
!define APP_EXE "SubashAI.exe"
!define UNINST_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\SubashAI"

Name "${APP_NAME} ${APP_VERSION}"
BrandingText "${APP_NAME} · ${APP_VERSION}"
OutFile "..\dist\SubashAI-Setup-${APP_VERSION}.exe"
InstallDir "$LOCALAPPDATA\Programs\Subash AI"
InstallDirRegKey HKCU "Software\SubashAI" "InstallDir"
RequestExecutionLevel user
SetCompressor /SOLID lzma
SetCompressorDictSize 32
SetDatablockOptimize on
CRCCheck on
XPStyle on

VIProductVersion "0.3.0.0"
VIAddVersionKey "ProductName" "${APP_NAME}"
VIAddVersionKey "ProductVersion" "${APP_VERSION}"
VIAddVersionKey "FileDescription" "Subash AI installer"
VIAddVersionKey "FileVersion" "0.3.0.0"
VIAddVersionKey "LegalCopyright" "MIT License"
VIAddVersionKey "CompanyName" "${APP_PUBLISHER}"

!define MUI_ABORTWARNING
!define MUI_ICON "..\assets\subash.ico"
!define MUI_UNICON "..\assets\subash.ico"
!define MUI_HEADERIMAGE
!define MUI_HEADERIMAGE_BITMAP "assets\header.bmp"
!define MUI_WELCOMEFINISHPAGE_BITMAP "assets\side.bmp"
!define MUI_WELCOMEPAGE_TITLE "Установка Subash AI"
!define MUI_WELCOMEPAGE_TEXT "Subash AI — маленькая нейросеть на чистом NumPy, которая продолжает текст в стиле Пушкина.$\r$\n$\r$\nУстановка займёт меньше минуты. Python и зависимости уже внутри."
!define MUI_FINISHPAGE_RUN "$INSTDIR\${APP_EXE}"
!define MUI_FINISHPAGE_RUN_TEXT "Запустить Subash AI"
!define MUI_FINISHPAGE_LINK "Репозиторий на GitHub"
!define MUI_FINISHPAGE_LINK_LOCATION "${APP_URL}"

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "..\LICENSE"
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

!insertmacro MUI_LANGUAGE "Russian"

LangString DESC_SecMain ${LANG_RUSSIAN} "Приложение Subash AI, ярлык в меню «Пуск» и деинсталлятор."
LangString DESC_SecDesktop ${LANG_RUSSIAN} "Дополнительный ярлык на рабочем столе."

Function .onInit
  SetShellVarContext current
  ; Закрываем запущенную копию, иначе нельзя заменить exe.
  nsExec::Exec 'taskkill /IM ${APP_EXE} /F'
FunctionEnd

Function un.onInit
  SetShellVarContext current
FunctionEnd

Section "Subash AI" SecMain
  SectionIn RO
  SetOutPath "$INSTDIR"
  File "..\dist\${APP_EXE}"
  File "..\LICENSE"
  File "..\README.md"
  WriteUninstaller "$INSTDIR\Uninstall.exe"

  CreateDirectory "$SMPROGRAMS\Subash AI"
  CreateShortcut "$SMPROGRAMS\Subash AI\Subash AI.lnk" "$INSTDIR\${APP_EXE}" "" "$INSTDIR\${APP_EXE}" 0
  CreateShortcut "$SMPROGRAMS\Subash AI\Удалить Subash AI.lnk" "$INSTDIR\Uninstall.exe"

  WriteRegStr HKCU "Software\SubashAI" "InstallDir" "$INSTDIR"
  WriteRegStr HKCU "${UNINST_KEY}" "DisplayName" "${APP_NAME}"
  WriteRegStr HKCU "${UNINST_KEY}" "DisplayVersion" "${APP_VERSION}"
  WriteRegStr HKCU "${UNINST_KEY}" "Publisher" "${APP_PUBLISHER}"
  WriteRegStr HKCU "${UNINST_KEY}" "URLInfoAbout" "${APP_URL}"
  WriteRegStr HKCU "${UNINST_KEY}" "DisplayIcon" "$INSTDIR\${APP_EXE}"
  WriteRegStr HKCU "${UNINST_KEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "${UNINST_KEY}" "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegStr HKCU "${UNINST_KEY}" "QuietUninstallString" '"$INSTDIR\Uninstall.exe" /S'
  WriteRegDWORD HKCU "${UNINST_KEY}" "NoModify" 1
  WriteRegDWORD HKCU "${UNINST_KEY}" "NoRepair" 1

  ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
  IntFmt $0 "0x%08X" $0
  WriteRegDWORD HKCU "${UNINST_KEY}" "EstimatedSize" "$0"
SectionEnd

Section /o "Ярлык на рабочем столе" SecDesktop
  CreateShortcut "$DESKTOP\Subash AI.lnk" "$INSTDIR\${APP_EXE}" "" "$INSTDIR\${APP_EXE}" 0
SectionEnd

Section "Uninstall"
  SetShellVarContext current
  Delete "$INSTDIR\${APP_EXE}"
  Delete "$INSTDIR\LICENSE"
  Delete "$INSTDIR\README.md"
  Delete "$INSTDIR\Uninstall.exe"

  Delete "$SMPROGRAMS\Subash AI\Subash AI.lnk"
  Delete "$SMPROGRAMS\Subash AI\Удалить Subash AI.lnk"
  RMDir "$SMPROGRAMS\Subash AI"
  Delete "$DESKTOP\Subash AI.lnk"

  DeleteRegKey HKCU "${UNINST_KEY}"
  DeleteRegKey HKCU "Software\SubashAI"
  RMDir "$INSTDIR"
SectionEnd

!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SecMain} $(DESC_SecMain)
  !insertmacro MUI_DESCRIPTION_TEXT ${SecDesktop} $(DESC_SecDesktop)
!insertmacro MUI_FUNCTION_DESCRIPTION_END
