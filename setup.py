from setuptools import setup, find_packages
import os

# Read requirements.txt for dependencies
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read README.md for long_description if available
long_description = ''
if os.path.exists('README.md'):
    with open('README.md', encoding='utf-8') as f:
        long_description = f.read()

setup(
    name='sentimentAnalysis',
    version='1.0.0',
    description='End-to-End Movie Review Sentiment Analysis with Flask UI',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=requirements,
    python_requires='>=3.8',
)