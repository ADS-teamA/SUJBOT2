"""
Setup script for Document Analyzer
"""

from setuptools import setup, find_packages
from pathlib import Path

# Načtení README pro dlouhý popis
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Načtení requirements
requirements = []
requirements_file = this_directory / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="document-analyzer",
    version="1.0.0",
    author="Document Analyzer Team",
    description="Pokročilý nástroj pro paralelní analýzu rozsáhlých dokumentů pomocí Claude Code SDK",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/document-analyzer",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={
        "": ["../prompts/*.md", "../examples/*.md"],
    },
    install_requires=requirements,
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "document-analyzer=document_analyzer:main",
            "doc-analyze=document_analyzer:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Legal Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    keywords="document analysis, claude, ai, nlp, parallel processing, legal tech",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/document-analyzer/issues",
        "Source": "https://github.com/yourusername/document-analyzer",
        "Documentation": "https://github.com/yourusername/document-analyzer/wiki",
    },
)