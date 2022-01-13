from cx_Freeze import setup, Executable

base = None

executables = [Executable("gui.py", base=base)]

packages = ["idna", "cv2", "numpy", "scipy", "os", "PySimpleGUI", "resolution"]
options = {
    'build_exe': {
        'packages':packages,
    },
}

setup(
    name = "<any name>",
    options = options,
    version = "<any number>",
    description = '<any description>',
    executables = executables
)