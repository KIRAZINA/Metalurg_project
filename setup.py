from setuptools import setup, find_packages

setup(
    name="test-metal",
    version="1.0.0",
    description="Linear Regression & Optimization Framework for analyzing physicochemical properties of materials",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/KIRAZINA/Metalurg_project",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.23.0,<2.0",
        "pandas>=1.5.0,<2.0",
        "scipy>=1.9.0,<1.13",
        "statsmodels>=0.13.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.6.0,<4.0",
        "openpyxl>=3.1.0,<4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
