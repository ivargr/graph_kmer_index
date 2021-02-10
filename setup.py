from setuptools import setup
from distutils.core import setup, Extension
#from Cython.Build import cythonize
#from distutils.extension import Extension
#from Cython.Distutils import build_ext

ext_modules=[
    Extension("graph_kmer_index.cython_kmer_index",
              ["graph_kmer_index/cython_kmer_index.pyx"],
              libraries=["m"],
              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ],
              extra_link_args=['-fopenmp']
              ),
    Extension("graph_kmer_index.cython_reference_kmer_index",
              ["graph_kmer_index/cython_reference_kmer_index.pyx"],
              libraries=["m"],
              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ],
              extra_link_args=['-fopenmp']
              )]

setup(name='graph_kmer_index',
      version='0.0.1',
      description='Graph Kmer Index',
      url='http://github.com/ivargr/graph_kmer_index',
      author='Ivar Grytten',
      author_email='',
      license='MIT',
      packages=["graph_kmer_index"],
      zip_safe=False,
      install_requires=['numpy', 'sortedcontainers', 'tqdm', 'biopython', 'numba', 'cython', 'SharedArray'],
      classifiers=[
            'Programming Language :: Python :: 3'
      ],
      entry_points={
            'console_scripts': ['graph_kmer_index=graph_kmer_index.command_line_interface:main']
      },
      #cmdclass = {"build_ext": build_ext},
      ext_modules = ext_modules
)
