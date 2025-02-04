import glob
import re
import setuptools
from typing import List


def get_version() -> str:
    """Get version from __init__.py without importing the package."""
    with open("poker_ai/__init__.py", "r") as f:
        content = f.read()
        version_match = re.search(r'^__version__ = ["\']([^"\']*)["\']', content, re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")


def get_scripts_from_bin() -> List[str]:
    """Get all local scripts from bin so they are included in the package."""
    return glob.glob("bin/*")


def get_package_description() -> str:
    """Returns a description of this package from the markdown files."""
    with open("README.md", "r") as stream:
        readme: str = stream.read()
    with open("HISTORY.md", "r") as stream:
        history: str = stream.read()
    return f"{readme}\n\n{history}"


def get_requirements() -> List[str]:
    """Returns all requirements for this package."""
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()
    return requirements


setuptools.setup(
    name="poker_ai",
    version=get_version(),
    author="Leon Fedden, Colin Manko",
    author_email="leonfedden@gmail.com",
    description="Open source implementation of a CFR based poker AI player.",
    long_description=get_package_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/fedden/poker_ai",
    packages=setuptools.find_packages(),
    install_requires=get_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    scripts=get_scripts_from_bin(),
    python_requires=">=3.7",
)
