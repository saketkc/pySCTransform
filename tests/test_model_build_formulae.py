from pysctransform.pysctransform import build_model_formula


class TestBuildModelFormulae:
    def test_no_batch_single_latent_var(self):
        formula = build_model_formula(['log10_umi'], batch_var=None)
        assert formula == "log10_umi"

    def test_no_batch_multiple_latent_vars(self):
        formula = build_model_formula(['log10_umi', 'percent_mt'], batch_var=None)
        assert formula == "log10_umi + percent_mt"

    def test_with_batch_single_latent_var(self):
        formula = build_model_formula(['log10_umi'], batch_var='slice')
        assert formula == "(log10_umi) : C(slice) + C(slice) - 1"

    def test_with_batch_multiple_latent_vars(self):
        formula = build_model_formula(['log10_umi', 'percent_mt'], batch_var='slice')
        assert formula == "(log10_umi + percent_mt) : C(slice) + C(slice) - 1"
