from setuptools import setup, Extension
import os

# Set the CXX environment variable to 'g++' before invoking setup
os.environ['CXX'] = 'g++'

ext_module = Extension(
    'rottnest.libindex',  # Change 'yourpackage' to your package name
    sources=['src/index.cc', 'src/compactor.cc', 'src/fm_index.cc', 'src/vfr.cc', 'src/kauai.cc', 'src/plist.cc'],
    language = "c++",
    include_dirs=['src'],
    library_dirs=[],
    libraries=['glog','divsufsort', 'aws-cpp-sdk-s3', 'aws-cpp-sdk-core', 'lz4', 'snappy', 'zstd'],
    extra_compile_args=['-O3', '-g', '-fPIC','-Wno-sign-compare', '-Wno-strict-prototypes', '-fopenmp', '-std=c++17'], 
    extra_link_args = ['-lgomp']
)

ext_module_rex = Extension(
    'rottnest.librex',  # Change 'yourpackage' to your package name
    # sources=['src/tokenize.cc'],
    sources=['src/rex.cc'],
    language = "c++",
    libraries=['zstd','glog'],
    extra_objects = [ 'vendored/Trainer.o', 'vendored/Compressor.o'], 
    extra_compile_args=[ '-O3', '-g', '-fPIC','-Wno-sign-compare', '-Wno-strict-prototypes', '-fopenmp', '-std=c++17'], 
    extra_link_args = ['-lgomp', '-l:libarrow.so', '-l:libparquet.so']
)

setup(
    name='rottnest',  # Change to your package name
    version='1.0.4',
    description='Description of your package',
    ext_modules=[ext_module, ext_module_rex],
    packages=['rottnest'],  # Change to your package name
    package_data={'rottnest': ['libindex.so', 'librex.so']},
    install_requires=[
            'typing_extensions',
            'getdaft>=0.1.20',
            'pyarrow>=7.0.0',
            'duckdb',
            'boto3',
            'pandas',
            'polars>=0.18.0', # latest version of Polars generally
            'sqlglot', # you will be needing sqlglot. not now but eventually
            'tqdm',
            ], # add any additional packages that 
    entry_points={
        "console_scripts": [
            "rottnest-search=rottnest.search:main",
            "rottnest-index=rottnest.index:main"
        ],
    }, 
)
