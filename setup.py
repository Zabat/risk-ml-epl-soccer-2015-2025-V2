"""
Setup script for EPL Betting Model package.

Install in development mode:
    pip install -e .

Install for production:
    pip install .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = (this_directory / "requirements.txt").read_text().strip().split('\n')
requirements = [r.strip() for r in requirements if r.strip() and not r.startswith('#')]

setup(
    name="epl-betting-model",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="ML system for predicting EPL home wins with betting backtest",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/epl-betting-model",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/epl-betting-model/issues",
        "Documentation": "https://github.com/yourusername/epl-betting-model#readme",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment",
    ],
    package_dir={"": "."},
    packages=find_packages(where="."),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "epl-predict=src.predictor:main",
            "epl-backtest=src.backtest:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
)
