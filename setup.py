import os
from setuptools import find_packages, setup

with open(os.path.join(os.path.dirname(__file__), "README.md")) as f:
    long_description = f.read()

setup(
    name="ptgnn",
    packages=find_packages(),
    license="MIT",
    package_dir={"ptgnn": "ptgnn"},
    test_suite="ptgnn.tests",
    python_requires=">=3.6.1",
    description="Graph Neural Network library for PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Deep Procedural Intelligence",
    package_data={"ptgnn": ["py.typed"]},
    install_requires=[
        "dpu-utils>=0.2.17",
        "jellyfish",
        "numpy",
        "torch-scatter>=2.0.4",
        "torch>=1.4.0",
        "tqdm",
        "typing-extensions",
    ],
    extras_require={"dev": ["black", "isort", "pre-commit"], "aml": ["azureml"]},
    setup_requires=["setuptools_scm"],
    url="https://github.com/microsoft/ptgnn/",
    project_urls={
        "Bug Tracker": "https://github.com/microsoft/ptgnn/issues",
        "Documentation": "https://github.com/microsoft/ptgnn/tree/master/docs",
        "Source Code": "https://github.com/microsoft/ptgnn",
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Topic :: Software Development :: Libraries",
        "Typing :: Typed",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    zip_safe=False,
)
