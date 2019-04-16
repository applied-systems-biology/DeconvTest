from setuptools import setup


setup(
    name='deconvtest',    # This is the name of your PyPI-package.
    version='1.0',                          # python versioneer
    url="https://github.com/applied-systems-biology/DeconvTest",
    author="Anna Medyukhina",
    packages=['DeconvTest', 'DeconvTest.classes', 'DeconvTest.batch', 'DeconvTest.fiji', 'DeconvTest.modules'],
    scripts=['DeconvTest/scripts/run_simulation.py'],
    author_email='anna.medyukhina@gmail.com',
    license='BSD-3-Clause',
    package_data={'': ['fiji_path']},
    include_package_data=True,

    install_requires=[
        'scikit-image',
        'pandas',
        'numpy',
        'seaborn',
        'scipy',
        'ddt'
      ],
 )
