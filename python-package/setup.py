# -*- coding:utf-8 -*-
import os
import sys
import shutil
import subprocess
import logging
from setuptools import find_packages, setup, Extension
from setuptools.command import build_ext, sdist, install_lib


CURRENT_DIR = os.path.dirname(os.path.expanduser(os.path.abspath(__file__)))
BUILD_DIR: str = "none"
REMOVE_DIR = set()
REMOVE_FILE = set()


def clean_all():
    # Clean up global temporary files and directories at the end of the run
    for path in REMOVE_DIR:
        shutil.rmtree(path)
    for path in REMOVE_FILE:
        os.remove(path)


def lib_name():
    # Get the name of the library that depends on the platform
    return "lib_utboost.dll" if os.name == "nt" else "lib_utboost.so"


def copy_project_src(project_dir, target_dir):
    # Copy c++ source code
    for dir_name in ("src", "include"):
        dst = os.path.join(target_dir, dir_name)
        shutil.copytree(os.path.join(project_dir, dir_name), dst)
        REMOVE_DIR.add(dst)

    for file_name in ("CMakeLists.txt", "LICENSE", "NOTICE-Third-Party"):
        dst = os.path.join(target_dir, file_name)
        shutil.copy(os.path.join(project_dir, file_name), dst)
        REMOVE_FILE.add(dst)


class CMakeExtension(Extension):

    def __init__(self, name):
        super(CMakeExtension, self).__init__(name, sources=[])


class CMakeBuild(build_ext.build_ext):

    logger = logging.getLogger("UTBoost build_ext")

    def build_extension(self, ext: Extension):
        if isinstance(ext, CMakeExtension):
            # check cmake
            try:
                _ = subprocess.run(["cmake", "--version"], check=True)
            except OSError:
                self.logger.error("CMake is not installed. "
                                  "Please install CMake and make sure it is added to the system\"s PATH.")
                sys.exit(1)

            env = os.environ.copy()
            global BUILD_DIR
            BUILD_DIR = os.path.abspath(self.build_temp)
            shutil.rmtree(BUILD_DIR, ignore_errors=True)
            os.makedirs(BUILD_DIR, exist_ok=True)
            # check cmake_lists
            cmake_lists_dir = os.path.join(CURRENT_DIR, os.path.pardir)
            if not os.path.isfile(os.path.join(cmake_lists_dir, "CMakeLists.txt")):
                cmake_lists_dir = os.path.join(CURRENT_DIR, "utboost")
                if not os.path.isfile(os.path.join(cmake_lists_dir, "CMakeLists.txt")):
                    raise RuntimeError("Can\"t find CMakeLists.txt.")
            # compile project
            cmake_args = ["cmake", cmake_lists_dir, "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + BUILD_DIR]
            self.logger.info("Starting to compile with CMake.")
            self.logger.info("Configure CMAKE_SOURCE_DIR=%s" % cmake_lists_dir)
            self.logger.info("Configure CMAKE_LIBRARY_OUTPUT_DIRECTORY=%s" % BUILD_DIR)
            if os.name == "nt":  # windows
                status = 0
                for gen in ("Visual Studio 17 2022", "Visual Studio 16 2019",
                            "Visual Studio 15 2017", "Visual Studio 14 2015",
                            "MinGW Makefiles"):
                    try:
                        command = cmake_args + ["-G", gen]
                        if gen.startswith("Visual"):
                            command += ["-A", "x64"]
                        subprocess.check_call(command, cwd=BUILD_DIR, env=env)
                        self.logger.info("%s is used for building Windows distribution.", gen)
                        status = 1
                        break
                    except subprocess.CalledProcessError:
                        shutil.rmtree(BUILD_DIR, ignore_errors=True)
                        os.makedirs(BUILD_DIR, exist_ok=True)
                        continue
                if status == 0:
                    raise RuntimeError("Please install Visual Studio or MinGW first "
                                       "and make sure it is added to the system\"s PATH.")

            else:  # linux
                subprocess.check_call(cmake_args, cwd=BUILD_DIR, env=env)
            # build
            subprocess.check_call(["cmake", "--build", ".", "--config", "Release"], cwd=BUILD_DIR, env=env)
        else:
            super().build_extension(ext)


class CustomSdist(sdist.sdist):

    logger = logging.getLogger("UTBoost sdist")

    def run(self):
        self.logger.info("Copy c++ source into Python directory.")
        copy_project_src(os.path.join(CURRENT_DIR, os.path.pardir), os.path.join(CURRENT_DIR, "utboost"))
        super().run()


class CustomInstallLib(install_lib.install_lib):

    logger = logging.getLogger("UTBoost install_lib")

    def install(self):
        outfiles = super().install()
        lib_dir = os.path.join(self.install_dir, "utboost", "lib")
        if not os.path.exists(lib_dir):
            os.mkdir(lib_dir)
        dst = os.path.join(self.install_dir, "utboost", "lib", lib_name())

        assert BUILD_DIR != "none"
        pre_lib_dir = os.path.join(CURRENT_DIR, os.path.pardir, "lib")
        build_dir = os.path.join(BUILD_DIR)

        if os.path.exists(os.path.join(pre_lib_dir, lib_name())):  # built by CMake directly
            src = os.path.join(pre_lib_dir, lib_name())
        else:  # built by setup.py
            status = 0
            src = ""
            for path in [os.path.join(build_dir, path + lib_name()) for path in ("", "Release/", "windows/x64/DLL/")]:
                if os.path.exists(path):
                    src = path
                    status = 1
                    break
            if status == 0:
                raise RuntimeError("Please build library first.")
        self.logger.info("Installing shared library: %s", src)
        dst, _ = self.copy_file(src, dst)
        outfiles.append(dst)
        return outfiles


if __name__ == "__main__":

    with open(os.path.join(CURRENT_DIR, "README.md"), encoding="utf-8") as fd:
        description = fd.read()

    logging.basicConfig(level=logging.INFO)

    setup(
        name="utboost",
        version=open(os.path.join(CURRENT_DIR, "utboost/VERSION")).read().strip(),
        description="UTBoost Python Package",
        long_description=description,
        long_description_content_type="text/markdown",
        install_requires=[
            "numpy",
            "pandas"
        ],
        python_requires=">=3.6",
        author="Junjie Gao",
        author_email="gaojunjie10@jd.com",
        zip_safe=False,
        packages=find_packages(),
        license="MIT",
        ext_modules=[CMakeExtension(name="lib_utboost")],
        classifiers=["License :: OSI Approved :: MIT License",
                     "Development Status :: 5 - Production/Stable",
                     "Natural Language :: English",
                     "Intended Audience :: Science/Research",
                     "Operating System :: Microsoft :: Windows",
                     "Operating System :: Unix",
                     "Programming Language :: Python",
                     "Programming Language :: Python :: 3",
                     "Programming Language :: Python :: 3.6",
                     "Programming Language :: Python :: 3.7",
                     "Programming Language :: Python :: 3.8",
                     "Topic :: Scientific/Engineering :: Artificial Intelligence"],
        cmdclass={
            "build_ext": CMakeBuild,
            "sdist": CustomSdist,
            "install_lib": CustomInstallLib
        },
        include_package_data=True,
        url="https://github.com/jd-opensource/UTBoost",
    )

    clean_all()
