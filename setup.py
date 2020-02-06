import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="chatty", 
    version="0.0.1",
    long_description=long_description,
    long_description_content_type="text/markdown"
)