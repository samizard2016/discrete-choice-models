from cx_Freeze import setup, Executable
import os
import sys
import scipy

# Define the virtual environment path
VENV_PATH = r"C:\works 2025 July plus\discrete choice models\Schneider\env"
PYTHON_INSTALL_DIR = os.path.dirname(os.path.dirname(os.__file__))  # Points to virtual environment
os.environ['TCL_LIBRARY'] = os.path.join(PYTHON_INSTALL_DIR, 'tcl', 'tcl8.6')
os.environ['TK_LIBRARY'] = os.path.join(PYTHON_INSTALL_DIR, 'tcl', 'tk8.6')

# Include SciPy library directory for DLLs
scipy_path = os.path.join(VENV_PATH, "Lib", "site-packages", "scipy")
scipylibs_path = os.path.join(VENV_PATH, "Lib", "site-packages", "scipy.libs")
# scipy_libs_path = os.path.join(scipy_path, "scipy.libs")

# Define files to include
files = {
    "include_files": [
        # Python and Tcl/Tk DLLs
        os.path.join(PYTHON_INSTALL_DIR, "DLLs", "tcl86t.dll"),
        os.path.join(PYTHON_INSTALL_DIR, "DLLs", "tk86t.dll"),
        os.path.join(PYTHON_INSTALL_DIR, "python3.dll"),
        os.path.join(PYTHON_INSTALL_DIR, "vcruntime140.dll"),
        # Note: Corrected typo from "msvcp140.dlx" to "msvcp140.dll"
        "msvcp140.dlx",
        "vcomp140.dlx",
        # Resource files
        "home.jpeg",
        "model.jpeg",
        "dashboard.jpeg",
        "simulatorx.jpeg",
        "model.ico",
        "data_final.xlsx",
        "profiles.xlsx",
        "groups_final.xlsx",
        "scenarios.xlsx",
        "schneider_choice_model_data.pkl",
        "schneider_choice_model_model.pkl",
        # PySide6 DLLs
        os.path.join(VENV_PATH, "Lib", "site-packages", "PySide6", "Qt6Core.dll"),
        os.path.join(VENV_PATH, "Lib", "site-packages", "PySide6", "Qt6Gui.dll"),
        os.path.join(VENV_PATH, "Lib", "site-packages", "PySide6", "Qt6Widgets.dll"),
        os.path.join(VENV_PATH, "Lib", "site-packages", "shiboken6", "shiboken6.abi3.dll"),
        # PySide6 plugins
        (os.path.join(VENV_PATH, "Lib", "site-packages", "PySide6", "plugins"), "PySide6/plugins"),
        # Image files for dashboard
        ("images/utilities.png", "images/utilities.png"),
        ("images/feature_importance_dcm.png", "images/feature_importance_dcm.png"),
        ("images/price_elasticity.png", "images/price_elasticity.png"),
        ("images/profile_shares_line.png", "images/profile_shares_line.png"),
         ("images/value_shares_line.png", "images/value_shares_line.png"),
        ("images/utilities_with_uncertainty_bdcm.png", "images/utilities_with_uncertainty_bdcm.png"),
        ("images/wtp_analysis_enhanced_bdcm.png", "images/wtp_analysis_enhanced_bdcm.png"),
        # SciPy library directory (includes DLLs like libopenblas.dll)
        (scipy_path, "scipy"),
        # (scipylibs_path, "scipy.libs")
    ],
    "includes": [
        "pandas",
        "numpy",
        "atexit",
        "numpy.lib.format",
        "multiprocessing",
        "PySide6",
        "shiboken6",
        "scipy"      
    ],
    "excludes": [
        "boto.compat.sys",
        "boto.compat._sre",
        "boto.compat._json",
        "boto.compat._locale",
        "boto.compat._struct",
        "boto.compat.array"
    ],
    "packages": [
        "pandas",
        "numpy",
        "matplotlib",
        "PySide6",
        "shiboken6",
        "scipy",
        "openpyxl"
    ]
}

# Check for missing files and raise error to stop the build
for file_path in files["include_files"]:
    if isinstance(file_path, tuple):
        file_path = file_path[0]
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing required file: {file_path}")

base = "Win32GUI" if sys.platform == "win32" else None

executables = [
    Executable(
        script="kdcmx.py",
        shortcut_name="Kantar Discrete Choice Model 0.1.3",
        shortcut_dir="DesktopFolder",
        icon="model.ico",
        base=base
    )
]

setup(
    name="Kantar Discrete Choice Model 0.1.3",
    author="Samir Paul",
    options={"build_exe": files},
    version="0.1.3",
    description="Kantar Discrete Choice Model 0.1.3",
    executables=executables
)