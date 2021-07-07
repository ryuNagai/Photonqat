import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    install_requires = list(map(str.strip, f))

setuptools.setup(
    name="photonqat", # Replace with your own username
    version="0.2.3",
    author="The Photonqat Developers",
    author_email="nagairic@gmail.com",
    description="Library for photonic continuous variable quantum programming",
    #long_description=long_description,
    #long_description_content_type="text/markdown",
    url="https://github.com/Blueqat/Photonqat",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
