=======
History
=======

0.1.2 (Unreleased)

Bugs:
* Fix for "TypeError: unsupported operand type(s) for -: 'IntVector' and 'int'" `#8 <https://github.com/saketkc/pySCTransform/pull/8>`_
* Fixed theta_ml to match scTransform implementation - 9 iterations by default,
* Silverman's bandwidth selection,


Features:
* Faster implementation of robust_scale_binned - ~25%.
* Add "log_umi" attribute to match R's.
* Handle sparse matrix in row_gmean, get_model_params_allgene_glmgp and pearson_residual.
* Add cell_attr_extra and batch_var support.
* Add tests comparing R with Python implementation.

