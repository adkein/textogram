from setuptools import setup

setup(
        name='textogram',
        version='0.1',
        description='Text-based histograms',
        author='Carl Smith',
        author_email='adkein@gmail.com',
        url='https://github.com/adkein/textogram',
        package_dir={'': 'src'},
        py_modules=['textogram'],
        scripts=['src/textogram'],
        install_requires=[
            'matplotlib',
        ],
        )
