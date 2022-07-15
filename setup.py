from setuptools import setup, find_packages

if __name__ == "__main__":
    hard_dependencies = ('numpy', 'scipy')
    install_requires = list()
    with open('requirements.txt', 'r') as fid:
        for line in fid:
            req = line.strip()
            for hard_dep in hard_dependencies:
                if req.startswith(hard_dep):
                    install_requires.append(req)

    setup(
        name="fastrakdigitizer",
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
        packages=find_packages(),
        include_package_data=True,
        install_requires=install_requires,
        entry_points={"console_scripts": ["fastrakdigitizer=fastrakdigitizer.main:main"]},
    )
