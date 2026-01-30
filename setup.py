#!/usr/bin/env python3
"""
SenseRepair: Sense Discovery via Simulated Self-Repair
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="sense-repair",
    version="0.2.0",
    author="Kow Kuroda & Claude",
    author_email="kow.k@ks.kyorin-u.ac.jp",
    description="Discover and disambiguate word senses in static embeddings via simulated self-repair",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kow-k/sense-repair",
    project_urls={
        "Bug Tracker": "https://github.com/kow-k/sense-repair/issues",
        "Documentation": "https://github.com/kow-k/sense-repair#readme",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black",
            "flake8",
        ],
    },
    entry_points={
        "console_scripts": [
            "sense-repair=sense_repair:main",
        ],
    },
    keywords=[
        "nlp",
        "word-embeddings",
        "word-sense-disambiguation",
        "polysemy",
        "glove",
        "word2vec",
        "semantic-similarity",
        "sense-discovery",
        "distributional-semantics",
    ],
)
