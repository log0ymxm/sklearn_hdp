import os
from distutils.core import setup
from Cython.Build import cythonize

version ='0.1'

here = os.path.abspath(os.path.dirname(__file__))
try:
    README = open(os.path.join(here, 'README.rst')).read()
    CHANGES = open(os.path.join(here, 'CHANGES.txt')).read()
except IOError:
    README = CHANGES = ''

install_requires = [
    'scikit-learn',
    ]

tests_require = [
    'mock',
    'pytest',
    'pytest-cov',
    ]

docs_require = [
    'Sphinx',
    ]

setup(
    name='hdp',
    version=version,
    description='Online Hierarchical Dirichlet Process that follows the SKLearn interface',
    long_description='\n\n'.join([README, CHANGES]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.4",
    ],
    keywords='',
    author='Paul English',
    author_email='paul@onfrst.com',
    url='https://github.com/log0ymxm/sklearn_hdp',
    license='MIT',
    packages=['hdp'],
    install_requires=install_requires,
    extras_require={
        'testing': tests_require,
        'docs': docs_require,
    },
    ext_modules = cythonize("hdp/hdp.pyx"),
)
