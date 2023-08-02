from setuptools import find_packages, setup

PACKAGE = "python_package_template"

setup(
    name="python-package-template",
    version="0.0.1",
    packages=find_packages(include=PACKAGE),
    install_requires=[
        "pandas",
        "numpy",
        "python-dotenv",
        "requests",
        "loguru",
    ],
    extras_require={
        "dev": [
            "black",
            "isort",
            "pycln",
            "pytest",
            "pytest-cov",
            "pytest-mock",
            "radon",
            "codespell",
            "pre-commit",
            "pyright",
            "pylint",
            "pandas-stubs",
        ]
    },
)
