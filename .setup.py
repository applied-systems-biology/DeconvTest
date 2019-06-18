from setuptools import setup


setup(
    name='deconvtest',
    version='3.0',
    url="https://applied-systems-biology.github.io/DeconvTest/",
    download_url="https://github.com/applied-systems-biology/DeconvTest",
    author="Anna Medyukhina",
    author_email='anna.medyukhina@gmail.com',
    packages=['DeconvTest', 'DeconvTest.classes', 'DeconvTest.batch', 'DeconvTest.fiji', 'DeconvTest.modules'],
    scripts=['DeconvTest/scripts/run_simulation.py'],
    license='BSD-3-Clause',
    package_data={'': ['fiji_path']},
    include_package_data=True,
    test_suite='DeconvTest.tests',

    install_requires=[
        'scikit-image',
        'pandas',
        'numpy',
        'seaborn',
        'scipy',
        'ddt',
        'helper_lib',
      ],
    dependency_links=[
        "https://github.com/applied-systems-biology/HelperLib/releases/",
    ],
 )

