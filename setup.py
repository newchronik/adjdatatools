from setuptools import setup, find_packages

setup(
    name='adjdatatools',
    version='0.3.2',
    packages=find_packages(),
    url='https://github.com/newchronik/adjdatatools',
    license='MIT',
    keywords='data preprocessing scaler',
    author='Anton Kvasnitsa',
    author_email='anton.kvasnitsa@gmail.com',
    description='This library contains adjusted tools for data preprocessing and working with mixed data types.',
    install_requires=[
        'numpy',
        'pandas'
    ],
    include_package_data=True,
)
