# -*- coding:utf-8 -*-
import os
import sys
from typing import List


class UTBoostLibraryNotFound(Exception):
    pass


def find_lib_path() -> List[str]:
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    dll_path = [curr_path,
                os.path.join(curr_path, "lib"),
                os.path.join(sys.prefix, "utboost"),
                ]

    if os.name == "nt":
        dll_path.append(os.path.join(curr_path, "../../windows/x64/Dll/"))
        dll_path.append(os.path.join(curr_path, "./windows/x64/Dll/"))
        dll_path.append(os.path.join(curr_path, "../../Release/"))
        dll_path.append(os.path.join(curr_path, "../../build/Release/"))
        dll_path = [os.path.join(p, "lib_utboost.dll") for p in dll_path]
    else:
        dll_path = [os.path.join(p, "lib_utboost.so") for p in dll_path]
    lib_path = [p for p in dll_path if os.path.exists(p) and os.path.isfile(p)]
    if not lib_path:
        dll_path = [os.path.realpath(p) for p in dll_path]
        raise UTBoostLibraryNotFound("Cannot find utboost library in following paths: " + ",".join(dll_path))
    return lib_path
