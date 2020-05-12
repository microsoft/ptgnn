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
        "dpu-utils>=0.2.15",
        "jellyfish",
        "numpy",
        "torch-scatter==2.0.4",
        "torch>=1.4.0",
        "tqdm",
        "typing-extensions",
    ],
    extras_require={"dev": ["black", "isort",]},
    setup_requires=["setuptools_scm"],
    url="https://deepproceduralintelligence.visualstudio.com/pt-gnn/",
    project_urls={
        "Bug Tracker": "https://deepproceduralintelligence.visualstudio.com/pt-gnn/_workitems/recentlyupdated/",
        "Documentation": "https://deepproceduralintelligence.visualstudio.com/pt-gnn/_git/pt-gnn?path=%2FREADME.md",
        "Source Code": "https://deepproceduralintelligence.visualstudio.com/pt-gnn/_git/pt-gnn",
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Topic :: Software Development :: Libraries",
        "Scientific/Engineering :: Artificial Intelligence",
        "Typing :: Typed",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    zip_safe=False,
)
