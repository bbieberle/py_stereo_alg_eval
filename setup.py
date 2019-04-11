from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='pystereoalgeval',
      version='0.2.3',
      description='Automated Stereo Algorithm Evaluation',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/dettoman/py_stereo_alg_eval',
      author='Benedikt Bieberle',
      author_email='benedikt.bieberle@tu-ilmenau.de',
      packages=find_packages(),
      classifiers=["Programming Language :: Python :: 3",
                   "Development Status :: 3 - Alpha",
                   "Environment :: Console",
                   "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
                   "Operating System :: OS Independent",
                   "Natural Language :: English",
                   "Natural Language :: German",
                   "Topic :: Software Development :: Testing"]
      )
