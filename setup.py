from setuptools import setup, find_packages

setup(
    name='didynet',
    version='1.0.0',
    description='DiDyNet: Differential Dynamic Network Inference',
    author='Zhe Liu',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scipy>=1.7.0',
        'statsmodels>=0.13.0',
        'dtaidistance>=2.3.0',
        'networkx>=2.6.0',
        'scikit-learn>=1.0.0',
        'tqdm>=4.62.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'joblib>=1.0.1'
    ],
    python_requires='>=3.8',
)
