from distutils.core import setup, Extension



setup(name='graph_kmer_index',
      version='0.0.3',
      description='Graph Kmer Index',
      url='http://github.com/ivargr/graph_kmer_index',
      author='Ivar Grytten',
      author_email='',
      license='MIT',
      packages=["graph_kmer_index"],
      zip_safe=False,
      install_requires=['numpy==1.20.3', 'sortedcontainers', 'tqdm', 'biopython', 'numba==0.54.1', 'cython', 'SharedArray', 'pyfaidx'],
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