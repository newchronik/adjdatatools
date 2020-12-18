from setuptools import setup, find_packages

setup(
    name='smart-data-tools',
    version='0.3.0',
    packages=find_packages(),
    url='https://github.com/newchronik/smart-data-tools',
    license='MIT',
    keywords='data preprocessing scaler',
    author='Anton Kvasnitsa',
    author_email='anton.kvasnitsa@gmail.com',
    description='This library contains adjusted tools for data preprocessing and working with mixed data types.',
    install_requires=[
        'numpy',
        'pandas',
        'math'
    ],
    include_package_data=True,
)
