import setuptools
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '1.2.5'
DESCRIPTION = 'To be added in the future'


setuptools.setup(
    name='ise_cdg_utility',
    version=VERSION,
    author="Ashkan Khademian",
    author_email="ashkan.khd.q@gmail.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=setuptools.find_packages(),
    include_package_data=True, 
    install_requires=[
        'torch>=2.0.0',
        'torchtext>=0.15.0',
        'torchmetrics>=1.2.0',
        'sentence-transformers>=2.5.0',
        "tqdm>=4.0.0",
        "bert-score==0.3.13",
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
    ]
)