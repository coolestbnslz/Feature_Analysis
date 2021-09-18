import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Feature_Analysis", # Replace with your own username
    version="0.0.1",
    author="Nikhil Bansal",
    author_email="bansalz1208@gmail.com",
    description="It performs feature analysis for data preprocessing or usage of data in Machine Learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/coolestbnslz/Feature_Analysis",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
