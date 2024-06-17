from setuptools import setup, find_packages

setup(
    name             = 'mdpl',
    version          = '1.0.0',
    description      = 'package for distribution',
    author           = 'KANG97',
    author_email     = 'ktt0570@gmail.com',
    url              = '',
    download_url     = '',
    install_requires = ['pandas', 'scikit-learn', 'matplotlib'],
	include_package_data=True,
	packages=find_packages(),
    keywords         = ['MEDICALDATAPIPELINE', 'medicaldatapipeline'],
    python_requires  = '>=3',
    zip_safe=False,
    classifiers      = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ]
) 