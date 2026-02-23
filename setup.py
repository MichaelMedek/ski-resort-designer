import pathlib

from setuptools import find_packages, setup

HERE = pathlib.Path(__file__).parent
version_file = HERE / "version.txt"

with open(version_file, "r", encoding="utf-8") as fh:
    version = fh.readlines()[-1].strip()

with open(HERE / "README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open(HERE / "requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#") and not line.startswith("-e")
    ]

setup(
    name="skiresort_planner",
    version=version,
    author="Michael Medek",
    author_email="michimedi@gmail.com",
    description="Design ski resorts on real terrain with an interactive map interface",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MichaelMedek/Ski-Resort-Planner",
    packages=find_packages(include=["skiresort_planner", "skiresort_planner.*"]),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    include_package_data=True,
    zip_safe=False,
)
