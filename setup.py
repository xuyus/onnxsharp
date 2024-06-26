import distutils.command.build
import os
import subprocess

import setuptools.command.build_py
import setuptools.command.develop
import setuptools.command.install
from setuptools import setup, find_packages


TOP_DIR = os.path.realpath(os.path.dirname(__file__))
SRC_DIR = os.path.join(TOP_DIR, "onnxsharp")

try:
    git_version = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=TOP_DIR)
        .decode("ascii")
        .strip()
    )
except (OSError, subprocess.CalledProcessError):
    git_version = None

with open(os.path.join(TOP_DIR, "VERSION")) as version_file:
    VersionInfo = version_file.readline().strip()


class build_py(setuptools.command.build_py.build_py):
    def run(self):
        setuptools.command.build_py.build_py.run(self)


class build(distutils.command.build.build):
    def run(self):
        self.run_command("build_py")


class develop(setuptools.command.develop.develop):
    VersionInfo = VersionInfo + "." + git_version[:8]

    def run(self):
        self.run_command("build")
        setuptools.command.develop.develop.run(self)


cmdclass = {
    "build_py": build_py,
    "build": build,
    "develop": develop,
}

setup(
    name="onnxsharp",
    version=VersionInfo,
    description="ONNX Sharp",
    cmdclass=cmdclass,
    packages=find_packages(),
    license="Apache License v2.0",
    author="xuyus",
    author_email="",
    url="https://github.com/xuyus/onnx-sharp",
    install_requires=["numpy>=1.14.1", "onnx>=1.4.1", "pytest", "black"],
    dependencies=["ai-edge-model-explorer"],
    entry_points={
        "console_scripts": [
            "onnx_summarize=onnxsharp.command_line:cli_onnx_summarize",
            "onnx_clip_subgraph=onnxsharp.command_line:cli_onnx_clip_subgraph",
            "onnx_get_nodes=onnxsharp.command_line:cli_onnx_get_nodes",
            "onnx_to_text=onnxsharp.command_line:cli_onnx_to_text",
        ],
    },
    scripts=[
        "bin/ort_get_peak_details",
        "bin/ort_get_peak_op_summary",
        "bin/ort_get_peak_summary",
        "bin/ort_get_peak_step",
        "bin/ort_parse_peak_summary",
        "bin/ort_scan_output_in_order",
        "bin/ort_parse_output_name",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
