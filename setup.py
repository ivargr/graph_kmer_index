from distutils.core import setup, Extension



setup(name='graph_kmer_index',
      version='0.0.22',
      description='Graph Kmer Index',
      url='http://github.com/ivargr/graph_kmer_index',
      author='Ivar Grytten',
      author_email='',
      license='MIT',
      packages=["graph_kmer_index"],
      zip_safe=False,
      install_requires=['numpy', 'sortedcontainers', 'tqdm', 'biopython', 'cython', 'SharedArray', 'pyfaidx',
                        'shared_memory_wrapper>=0.0.11', 'npstructures', 'bionumpy', 'obgraph'],
      classifiers=[
            'Programming Language :: Python :: 3'
      ],
      entry_points={
            'console_scripts': ['graph_kmer_index=graph_kmer_index.command_line_interface:main']
      },
)


""""
rm -rf dist
python3 setup.py sdist
twine upload --skip-existing dist/*

"""