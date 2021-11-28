from distutils.core import setup, Extension



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
)
