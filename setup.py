from setuptools import setup, find_packages

setup(
    name='tutor_helper',
    packages=find_packages(include=["tutor_helper*"]),
    version='0.1.0',
    install_requires=[
        # Add dependencies here
    ],
    author='Yvette',
    author_email='tsaimc2@gmail.com',
    description='A tutor assistant.',
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.12',
)