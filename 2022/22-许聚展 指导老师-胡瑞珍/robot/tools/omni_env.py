import os

"""
change the omniverse_path to your own, then open a terminal: 

mkdir .vscode & python omni_env.py > .vscode/settings.json

"""

omniverse_path = "C:/omniverse/pkg/isaac_sim-2022.1.1"

python_path = os.path.join( omniverse_path, "kit/python/python.exe" ).replace("\\", "/" )

def log_import_path(path, ret=""):
    folders = os.listdir(path)
    for f in folders:
        ret += "        \"" + os.path.join(path, f).replace("\\", "/" ) + "\",\n"
    return ret


# add omni path
path = os.path.join( omniverse_path, "kit/extscore")
ret = log_import_path(path)

path = os.path.join( omniverse_path, "exts")
ret = log_import_path(path, ret)

path = os.path.join( omniverse_path, "kit/extsphysics")
ret = log_import_path(path, ret)
ret += "        \"" + os.path.join( omniverse_path, "kit/extsphysics/omni.usd.schema.physx/pxr").replace("\\", "/" ) + "\",\n"

ret += "        \"" + os.path.join( omniverse_path, "kit/plugins/bindings-python").replace("\\", "/" ) + "\",\n"

# add pip-module path, like numpy
ret += "        \"" + os.path.join( omniverse_path, "kit/extscore/omni.kit.pip_archive/pip_prebundle").replace("\\", "/" ) + "\""

# set json str
final_ret  = '''
{
    "python.defaultInterpreterPath": "%s",
    "python.autoComplete.extraPaths": [
%s
    ],
    "python.analysis.extraPaths": [
%s
    ]
}
''' % (
    python_path,
    ret,
    ret,
)

print(final_ret)
