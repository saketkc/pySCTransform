=======
History
=======

0.1.2 (Unreleased)

Bugs:
* Fix for "TypeError: unsupported operand type(s) for -: 'IntVector' and 'int'" `#8 <https://github.com/saketkc/pySCTransform/pull/8>`_
* Fix broken syntax in GLM design matrix code generation - where each batch can have its own intercept and slope.
* Fix fit.py returning a single mu value - generating incorrect theta estimates,
* Fixed theta_ml to match SCTransform implementation - 9 iterations by default,
* Use Silverman's bandwidth selection to be more like R SCTransform.

Features:
* 25% faster implementation of robust_scale_binned.
* Add "log_umi" attribute to match R's.
* Handle sparse matrix in row_gmean, get_model_params_allgene_glmgp and pearson_residual.
* Add cell_attr_extra and batch_var support.
* Add tests comparing R with Python implementation.

