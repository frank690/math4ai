# This is the setup.py which is used by setuptools (https://setuptools.readthedocs.io/en/latest/)
# to install your package

import re

# -*- coding: utf-8 -*-
import setuptools

with open("CHANGELOG.md", "r") as fh:
    changelog = fh.read().splitlines()

compiler = re.compile(pattern=r"^\s*version\s+\d+(\.\d+)*\s*$", flags=re.IGNORECASE)
raw_changelog_version = list(filter(compiler.match, changelog))[0]
changelog_version = re.sub(
    pattern=r"^\s*version\s+",
    repl="",
    string=raw_changelog_version,
    flags=re.IGNORECASE,
)

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("dev-requirements.txt", "r") as fh:
    dev_requirements = fh.read().splitlines()

with open("requirements.txt", "r") as fh:
    requirements = fh.read().splitlines()

requirements = [req for req in requirements if not req.lower().startswith("pytest")]

setuptools.setup(
    name="math4ai",
    version=changelog_version,
    author="Frank Eschner",
    author_email="frank.eschner@hhu.de",
    description="My personal sklearn rebuild to get a better understanding of how things work",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/frank690/math4ai/",
    packages=setuptools.find_packages(
        exclude=[
            "dist",
            "build",
            "tests",
            "tests.*",
            "*.tests.*",
            "*.tests",
            "docs",
            "venv",
        ]
    ),
    include_package_data=True,
    setup_requires=["cython"],
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
    },
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Operating System :: Microsoft :: Windows ",
        "Operating System :: Unix",
    ],
    python_requires=">=3.9",
)
