import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tradebot",
    version="0.1",
    author="Benjamin Biering",
    author_email="benjamin.biering@gmail.com",
    description="An automated AI-based trading bot",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BBiering/tradeBot",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[

    ]
)