from setuptools import setup, find_packages

setup(
    name="brain_tumor_api",
    version="0.1",
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines(),
    include_package_data=True,
)
