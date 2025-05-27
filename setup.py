from setuptools import setup, Extension
import pybind11
import platform

# check the os and decide the extra_link_args
EXTRA_LINK_ARGS = []
if platform.system() == "Darwin":
    EXTRA_LINK_ARGS = ["-undefined", "dynamic_lookup"]
else:
    EXTRA_LINK_ARGS = ["-shared"]

# Extension module of C++
ext_modules = [
    Extension(
        name="ph_opt.ph_compute.bin.rips_cpp",
        sources=["ph_opt/ph_compute/ph_cpp_library_pybind.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",

        # g++-14 -O3 -Wall -std=c++17 -fPIC
        extra_compile_args=["-O3", "-Wall", "-std=c++17", "-fPIC"],

        # extra_link_args
        extra_link_args=EXTRA_LINK_ARGS,
    ),
]

setup(
    name="ph_opt",
    version="1.0.1",
    description="Library for optimizing persistent homology",
    packages=[
        "ph_opt",
        "ph_opt.data",
        "ph_opt.optimizer",
        "ph_opt.ph_compute",
        "ph_opt.ph_compute.bin",
        "ph_opt.ph_grad",
        "ph_opt.ph_loss",
        "ph_opt.scheduler",
        "ph_opt.utils",
    ],
    ext_modules=ext_modules,
)
