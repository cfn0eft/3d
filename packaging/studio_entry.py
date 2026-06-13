"""配布版 PoseLab Studio (exe) のエントリポイント。

PyInstaller でこのスクリプトを固める。1 つの exe が 3 役を兼ねる:

- 引数なし / serve オプション : Web GUI サーバーを起動してブラウザを開く
- ``--cli <args...>``         : poselab CLI として動く (GUI サーバーが
                                 ジョブ実行のサブプロセスとして自分自身を
                                 この形で起動する)
- ``--selftest``              : 同梱物の自己診断 (CI のスモークテスト用)
"""

from __future__ import annotations

import multiprocessing
import sys


def selftest() -> int:
    """同梱ライブラリの読み込みと GUI 資材の生成を確認する。"""
    import poselab

    print(f"poselab   : {poselab.__version__}")
    failures = 0

    def check(label, fn):
        nonlocal failures
        try:
            print(f"{label:10s}: {fn()}")
        except Exception as exc:  # noqa: BLE001 - 診断用
            failures += 1
            print(f"{label:10s}: NG ({exc})")

    check("python", lambda: sys.version.split()[0])
    check("numpy", lambda: __import__("numpy").__version__)
    check("opencv", lambda: __import__("cv2").__version__)
    check("mediapipe", lambda: __import__("mediapipe").__version__)

    def torch_info():
        import torch

        cuda = torch.cuda.is_available()
        detail = torch.cuda.get_device_name(0) if cuda else "CPU のみ"
        return f"{torch.__version__} (CUDA: {detail})"

    check("torch", torch_info)
    check("mmengine", lambda: __import__("mmengine").__version__)
    check("mmcv", lambda: __import__("mmcv").__version__)
    check("mmdet", lambda: __import__("mmdet").__version__)
    check("mmpose", lambda: __import__("mmpose").__version__)
    check(
        "inferencer",
        lambda: __import__(
            "mmpose.apis.inferencers", fromlist=["Pose3DInferencer"]
        ).Pose3DInferencer.__name__,
    )

    from poselab.studio import build_app_js

    check("gui", lambda: f"app.js {len(build_app_js()):,} bytes")
    print("selftest: " + ("OK" if failures == 0 else f"NG ({failures} 件)"))
    return 0 if failures == 0 else 1


def main() -> int:
    multiprocessing.freeze_support()  # Windows の frozen 環境では必須
    # 日本語ログ / 診断出力が Windows コンソール (cp1252) で落ちないよう UTF-8 に
    from poselab.studio.server import force_utf8_stdio

    force_utf8_stdio()
    argv = sys.argv[1:]
    if argv[:1] == ["--cli"]:
        from poselab.cli import main as cli_main

        return cli_main(argv[1:])
    if argv[:1] == ["--selftest"]:
        return selftest()
    from poselab.studio.server import main_serve

    return main_serve(argv)


if __name__ == "__main__":
    raise SystemExit(main())
