from setuptools import setup, Extension
import pybind11
import numpy as np

cg_solver_module = Extension(
    'cg_solver_cpp',
    sources=['cg_solver.cpp'],
    include_dirs=[
        pybind11.get_include(),
        np.get_include()
    ],
    language='c++',
    extra_compile_args=['-O3', '-fopenmp', '-std=c++11'],
    extra_link_args=['-fopenmp'],
)

setup(
    name='cg_solver_cpp',
    version='1.0',
    description='Параллельный решатель СЛАУ методом сопряжённых градиентов',
    ext_modules=[cg_solver_module],
)