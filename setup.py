"""Setup file for the Insurance Prediction project."""

from setuptools import find_packages, setup

entry_point = "insurance-prediction = insurance_prediction.__main__:main"

with open("requirements.txt", encoding="utf-8") as f:
    requires = f.read().splitlines()

setup(
    name="insurance_prediction",
    version="0.1",
    packages=find_packages(where="src", exclude=["tests"]),
    package_dir={"": "src"},
    entry_points={"console_scripts": [entry_point]},
    install_requires=requires,
    extras_require={
        "docs": [
            "sphinx>=1.6.3, <2.0",
            "sphinx_rtd_theme==0.4.1",
            "nbsphinx==0.3.4",
            "nbstripout==0.3.3",
            "sphinx-autodoc-typehints==1.6.0",
            "sphinx_copybutton==0.2.5",
            "ipykernel>=5.3, <7.0",
        ]
    },
)
