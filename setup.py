from setuptools import setup

setup(name='pae',
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      description='Probabilistic Auto-Encoder',
      url='http://github.com/VMBoehm/PAE',
      author='Vanessa Martina Boehm',
      author_email='vboehm@berkeley.edu',
      license='Apache License 2.0',
      packages=['pae'],
      )
