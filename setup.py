from setuptools import setup, find_packages

setup(
    name='AISample',
    version='1.0.0',
    description='pubmed abstract text classification project',
    url='https://github.com/cjs220/pubmed_text_classification',
    author='AI Systems, University of Manchester',
    author_email='conor.sheehan-3@postgrad.manchester.ac.uk',
    license='MIT',
    packages=find_packages(),
    zip_safe=False,
    setup_requires=["nose"],
    tests_require=["nose", "coverage"]
)

