from setuptools import setup, find_packages

setup(name='fabrica',
      version='0.0.1',
      description='Xyla\'s data generation tool.',
      url='https://github.com/xyla-io/fabrica',
      author='Xyla',
      author_email='gklei89@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
        'pytest',
        'click',
      ],
      zip_safe=False)