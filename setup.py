import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="audax",
    version="0.0.4",
    author='Sarthak Yadav',
    description="audio ML for Jax",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SarthakYadav/audax",
    # package_dir={"": ""},
    packages=[
        "audax",
        "audax.commons",
        "audax.core",
        "audax.frontends",
        "audax.models",
        "audax.models.layers",
        "audax.transforms",
        "audax.training_utils",
        "audax.training_utils.data_v2",
    ],
    python_requires=">=3.7"
)
