from setuptools import setup, find_packages
setup(
    name='audiosample',
    version="2.2.6",
    packages=find_packages(),
    install_requires=[
        'numpy<=2.2.0',
    ],
    extras_require={
        #minimal for proper working with notebooks.
        'jupyter': ['jupyter', 'IPython', 'librosa', 'matplotlib',],
        #include torch dependencies
        'torch': ['torch>=1.5',],
        #everything but required for display and play.
        'noui': ['torch>=1.5', 'av>=12.3.0,<14',],
        #include av dependencies
        'av': ['av>=12.3.0,<14', ],
        #allow play.
        'play': ['pyaudio', ],
        #full install for development not testing.
        'all': ['jupyter', 'IPython', 'librosa', 'matplotlib', 'torch>=1.5', 'av>=12.3.0,<14'],

        #testing dependencies
        'tests': ['pytest', 'jupyter', 'IPython', 'librosa', 'matplotlib', 'scipy', 'pyaudio', 'torch>=1.5', 'av>=12.3.0,<14', 'fire==0.7.0',],
    },
    author='Nir Krakowski',
    author_email='nir@deepdub.ai',
    description='AudioSample is an optimized numpy-like audio manipulation library, created for researchers, used by developers.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://deepdub.ai',
    project_urls={
        'Source': 'https://github.com/deepdub-ai/audiosample',
    },
    license="MIT",
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    include_package_data=True,
)
