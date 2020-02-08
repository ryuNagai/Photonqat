import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="photonqat", # Replace with your own username
    version="0.2.1-dev",
    author="The Photonqat Developers",
    author_email="nagairic@gmail.com",
    description="Library for photonic continuous variable quantum programming",
    #long_description=long_description,
    #long_description_content_type="text/markdown",
    url="https://github.com/Blueqat/Photonqat",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
