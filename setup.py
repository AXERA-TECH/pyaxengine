# Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
#
# This source file is the property of Axera Semiconductor Co., Ltd. and
# may not be copied or distributed in any isomorphic form without the prior
# written consent of Axera Semiconductor Co., Ltd.
#

from setuptools import setup

setup(
    name="axengine",
    version="0.1.2",
    classifiers=[
        "Development Status :: 1 - Alpha",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    packages=["axengine"],
    ext_modules=[],
    install_requires=["cffi>=1.0.0", "ml-dtypes>=0.1.0", "numpy>=1.22"],
    setup_requires=["cffi>=1.0.0", "ml-dtypes>=0.1.0", "numpy>=1.22"],
)
