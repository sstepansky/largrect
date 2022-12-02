import numpy as np
from setuptools import setup, Extension

setup(
    ext_modules=[
        Extension(
            name="largrect", sources=["largrect.c"], include_dirs=[np.get_include()]
        )
    ]
)
