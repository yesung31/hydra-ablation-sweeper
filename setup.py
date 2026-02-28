from setuptools import setup, find_namespace_packages

setup(
    name="hydra-ablation-sweeper",
    version="0.1.0",
    packages=find_namespace_packages(include=["hydra_plugins.*"]),
    install_requires=[
        "hydra-core>=1.1.0",
    ],
    include_package_data=True,
)
