==============
pySCTranscform
==============

SCTransform for Python - interfaces with `ScanPy <https://scanpy.readthedocs.io/en/stable/>`_

=============
Demo Notebook
=============

See `demo <notebooks/demo.ipynb>`_.


=============
Installation 
=============

Using conda
-------------

We recommend using `conda <https://docs.conda.io/en/latest/>`_ for installing pySCTransform.

::

    conda create -n pysct leidenalg scanpy jupyterlab
    conda activate pysct
    pip install git+https://github.com/saketkc/pysctransform.git@glmgp

If you would like to use `glmGamPoi <https://bioconductor.org/packages/release/bioc/html/glmGamPoi.html>`_, a faster estimator, ``rpy2`` and ``glmGamPoi`` need to be installed as well:

::

    conda create -n pysct leidenalg scanpy jupyterlab rpy2 bioconductor-glmgampoi
    conda activate pysct
    pip install git+https://github.com/saketkc/pysctransform.git@glmgp



