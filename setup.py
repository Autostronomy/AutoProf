from setuptools import setup, find_packages
import autoprof.__init__ as ap
import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="autoprof",
    version=ap.__version__,
    description="Fast, robust, deep isophotal solutions for galaxy images",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/Autostronomy/AutoProf",
    author=ap.__author__,
    author_email=ap.__email__,
    license="GPL-3.0 license",
    packages=find_packages(),
    package_data={"": ["*.png"]},
    include_package_data=True,
    install_requires=list(read("requirements.txt").split("\n")),
    entry_points={
        "console_scripts": [
            "autoprof = autoprof:run_from_terminal",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
    ],
)
