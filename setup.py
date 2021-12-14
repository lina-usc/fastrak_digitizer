from setuptools import setup

setup(
    name="fastrak_digitizer",
    version="0.1.0",
    description="EEG channel digitization application for the Polhemus Fastrak",
    #long_description=README,
    #long_description_content_type="text/markdown",
    url="https://github.com/christian-oreilly/fastrak_digitizer",
    author="Christian O'Reilly",
    author_email="christian.oreilly@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
    packages=["fastrakdigitizer"],
    include_package_data=True,
    install_requires=[
        'pyserial', 'PySide2', 'beepy', 'mne>=0.22', 'numpy', 'pandas', 'trimesh'
    ],
    entry_points={"console_scripts": ["fastrakdigitizer=fastrakdigitizer.main:main"]},
)
