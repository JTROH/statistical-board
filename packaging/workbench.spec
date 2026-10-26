# PyInstaller spec for "Statistical Workbench.app" (macOS). Build with
# packaging/build_mac_app.sh, not by hand -- it runs from the repo root.
# -*- mode: python -*-

from PyInstaller.utils.hooks import collect_submodules

ROOT = SPECPATH + "/.."

datas = [
    (ROOT + "/pdstat/static", "pdstat/static"),
    (ROOT + "/doe_advisor/static", "doe_advisor/static"),
    (ROOT + "/stat_board/static", "stat_board/static"),
    (ROOT + "/presets", "presets"),
    (ROOT + "/sample_data", "sample_data"),
]

a = Analysis(
    [ROOT + "/run_desktop.py"],
    pathex=[ROOT],
    datas=datas,
    # uvicorn picks its loop/protocol modules by name at runtime.
    hiddenimports=collect_submodules("uvicorn"),
    excludes=["tkinter", "IPython", "pytest"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Statistical Workbench", console=False)
coll = COLLECT(exe, a.binaries, a.datas, name="Statistical Workbench")
app = BUNDLE(
    coll,
    name="Statistical Workbench.app",
    bundle_identifier="com.jtroh.statistical-workbench",
    info_plist={
        "CFBundleShortVersionString": "0.1.0",
        "NSHighResolutionCapable": True,
        # The window talks to its own server on 127.0.0.1 over plain HTTP.
        "NSAppTransportSecurity": {"NSAllowsLocalNetworking": True},
    },
)
