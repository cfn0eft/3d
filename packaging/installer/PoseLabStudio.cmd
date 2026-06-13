@echo off
rem PoseLab Studio ランチャ。インストール先の専用 Python で GUI を起動する。
rem %~dp0 = この .cmd があるフォルダ (= インストール先)
"%~dp0env\Scripts\python.exe" -m poselab.studio %*
