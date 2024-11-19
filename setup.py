from setuptools import setup, find_packages

# Read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="clearbox-preprocessor",  
    version="0.9.7",  
    author="Dario Brunelli",
    author_email="dario@clearbox.ai",
    description="A polars based preprocessor for ML datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Clearbox-AI/preprocessor",  # Replace with your GitHub repository URL
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    classifiers=[
    ],
    python_requires='>=3.9',
    install_requires=[
                    'numpy<2.0.0',
                    'pandas',
                    'scikit-learn',
                    'polars<1.0.0',
                    'tsfresh'],
    extras_require={
        "dev": [
            "check-manifest",
            "pytest>=3.9",
        ],
        "test": [
            "coverage",
        ],
    },
    package_data={
    },
    entry_points={
    },
)