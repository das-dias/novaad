from setuptools import setup, find_packages
import os

# Read the contents of the README file for the long description
def read_readme():
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        return f.read()

setup(
    name="novaad",
    version="0.1.4",
    description="An Analog/Mixed-Signal IC design tool based on the Gm/Id method.",
    author="dasdias",
    author_email="das.dias@campus.fct.unl.pt",
    license="BSD-2",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords=["analog", "mixed-signal", "ic", "design", "gm/id", "eda-cad"],
    
    # Specify which packages to include
    packages=find_packages(exclude=["tests"]),
    
    # Specify files or directories to include in the distribution
    include_package_data=True,
    package_data={
        '': ['README.md', 'LICENSE', 'novaad/*.yaml', 'novaad/*.yml'],
    },
    
    # Include files from these directories
    data_files=[
        ('docs', ['docs/*']),
        ('tests', ['tests/*']),
    ],

    # Dependencies
    install_requires=[
        "pandas>=2.2.2,<3.0.0",
        "numpy>=2.1.0,<3.0.0",
        "PyYAML>=6.0.2,<7.0.0",
        "plotly>=5.24.0,<6.0.0",
        "docopt>=0.6.2,<0.7.0",
        "scipy>=1.14.1,<2.0.0",
        "pydantic>=2.8.2,<3.0.0",
        "toml>=0.10.2,<0.11.0",
        "confz>=2.0.1,<3.0.0"
    ],

    # Development dependencies
    extras_require={
        "dev": [
            "pytest>=8.3.2,<9.0.0",
            "pytest-benchmark>=4.0.0,<5.0.0",
            "poetry-setup>=0.3.6,<1.0.0",
            "poetry>=1.8.3,<2.0.0",
            "pytest-coverage"
        ]
    },

    # Entry point for the CLI
    entry_points={
        "console_scripts": [
            "novaad = novaad.__main__:main",
        ]
    },

    python_requires=">=3.11",
    include_package_data=True,  # If you have a MANIFEST.in file
    zip_safe=False,
)
