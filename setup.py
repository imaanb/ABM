import pathlib

from setuptools import find_packages, setup

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text(encoding="utf-8")

setup(
    name="sugarscape",
    version="0.1.0",
    description="Agent-based Sugarscape extension for fiscal-policy experiments",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="you@example.com",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "ipython",
        "joblib",
        "matplotlib",
        "mesa",
        "numpy",
        "pandas",
        "SALib",
        "seaborn",
        "solara",
        "tqdm",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
