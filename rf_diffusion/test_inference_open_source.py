def make_deterministic():
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_CBWR"] = "COMPATIBLE"
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["ATEN_CPU_CAPABILITY"] = "avx2"
    os.environ["ONEDNN_MAX_CPU_ISA"] = "AVX2"
    os.environ["ONEDNN_DEFAULT_FPMATH_MODE"] = "strict"

    import torch
    torch.use_deterministic_algorithms(True)

    import numpy as np
    import random
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_flush_denormal(True)
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    # Disable oneDNN/mkldnn fusion paths which can change numerics across CPUs
    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = False
make_deterministic()

def assert_deterministic():
    import os
    import torch

    expected_env = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_CBWR": "COMPATIBLE",
        "PYTHONHASHSEED": "0",
        "ATEN_CPU_CAPABILITY": "avx2",
        "ONEDNN_MAX_CPU_ISA": "AVX2",
        "ONEDNN_DEFAULT_FPMATH_MODE": "strict",
    }

    for var, expected in expected_env.items():
        actual = os.environ.get(var)
        assert actual == expected, (
            f"Environment variable {var!r} expected {expected!r} but got {actual!r}"
        )
        # Environment checks
    for var, expected in expected_env.items():
        actual = os.environ.get(var)
        assert actual == expected, (
            f"Environment variable {var!r} expected {expected!r} but got {actual!r}"
        )

    # PyTorch runtime checks
    assert torch.are_deterministic_algorithms_enabled(), "Deterministic algorithms not enabled"
    assert torch.get_num_threads() == 1, "Expected torch num threads to be 1"
    assert torch.get_num_interop_threads() == 1, "Expected torch interop threads to be 1"
assert_deterministic()

import torch
import unittest
import pytest
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
import hydra
from hydra import compose, initialize

import run_inference
# from functools import partial
# from rf2aa import tensor_util
# from rf_diffusion import inference
# from rf_diffusion.test_inference import infer, construct_conf

REWRITE = False
torch.use_deterministic_algorithms(True)

def infer(config_name, overrides):
    conf = construct_conf(config_name, overrides)
    run_inference.main(conf)
    p = Path(conf.inference.output_prefix + '_0-atomized-bb-True.pdb')
    return p, conf

def construct_conf(config_name, overrides):
    overrides = overrides + ['inference.cautious=False', 'inference.design_startnum=0']
    initialize(version_base=None, config_path="config/inference", job_name="test_app")
    conf = compose(config_name=f'{config_name}.yaml', overrides=overrides, return_hydra_config=True)
    # This is necessary so that when the model_runner is picking up the overrides, it finds them set on HydraConfig.
    HydraConfig.instance().set_config(conf)
    conf = compose(config_name=f'{config_name}.yaml', overrides=overrides)
    return conf

def run_inference_test(t, test_name, config_name, overrides, rewrite=False):
    overrides = overrides + [
        'inference.output_prefix=open_source_tests/' + test_name,
    ]
    pdb, conf = infer(config_name, overrides)
    print(f"Running inference test {test_name} with config {config_name} and overrides {overrides} ->\n{pdb}")
    # pdb_contents = inference.utils.parse_pdb(pdb)
    # cmp = partial(tensor_util.cmp, atol=0, rtol=0)
    # test_utils.assert_matches_golden(t, test_name, pdb_contents, rewrite=rewrite, custom_comparator=cmp)

AME_CASES = {
    "M0024_1nzy": [
        "inference.ckpt_path=REPO_ROOT/rf_diffusion/model_weights/RFD_140.pt",
        "++transforms.configs.CenterPostTransform.center_type='all'",
        "inference.contig_as_guidepost=True",
        "inference.state_dict_to_load=model_state_dict",
        "inference.str_self_cond=False",
        "inference.model_runner=NRBStyleSelfCond",
        "++idealization_metric_n_steps=50",
        "inference.idealize_sidechain_outputs=True",
        "++diffuser.rots.sample_schedule=normed_exp",
        "++diffuser.rots.exp_rate=10",
        "contigmap.reintersperse=True",
        "inference.guidepost_xyz_as_design_bb=[True]",
        "inference.input_pdb=benchmark/input/mcsa_41/M0024_1nzy.pdb",
        "inference.ligand='BCA'",
        "contigmap.contigs=['49,A64-64,21,A86-86,3,A90-90,23,A114-114,22,A137-137,7,A145-145,49']",
        "contigmap.contig_atoms=\"{'A64':'O,C','A86':'CB,CA,N,C','A90':'CE1,ND1,NE2,CG,CD2','A114':'N,CA','A137':'NE1,CD1,CE2,CG,CD2,CZ2','A145':'OD2,CG,CB,OD1'}\"",
        '++inference.partially_fixed_ligand={BCA:[C6B,C5B,C7B,C4B,O2B,C2B,C3B,C1B,S1P,O1B,C2P,C3P,N4P,C5P,C6P,O5P,C7P,N8P,C9P,CAP,O9P,CBP,OAP,CCP,CDP,CEP,O6A,P2A]}',
        "++inference.write_trajectory=True",
        "++inference.write_trb_indep=True",
        "++inference.write_trb_trajectory=True",
        "inference.num_designs=1",
        "inference.deterministic=True",
        "contigmap.intersperse='1-3'",
        "contigmap.length=5-20",
        "diffuser.T=2",
    ],
}

SIMPLE_CASES = {
    # TODO: add some of the cases from test_inference.py
    'basic': [
        'diffuser.T=1',
        'inference.num_designs=1',
        # f'inference.output_prefix=tmp/{test_name}_{output_suffix}',
        'inference.write_trajectory=True',
        'inference.write_trb_indep=True',
    ],
}

class HydraTest(unittest.TestCase):

    def setUp(self) -> None:
        # Some other test is leaving a global hydra initialized, so we clear it here.
        if hydra.core.global_hydra.GlobalHydra().is_initialized():
            hydra.core.global_hydra.GlobalHydra().clear()
        return super().setUp()

    def tearDown(self):
        hydra.core.global_hydra.GlobalHydra.instance().clear()

class TestInferenceOpenSource(HydraTest):
    @pytest.mark.open_source
    def test_ame_inference(self):
        '''
        Tests that running inference with the AME case M0024_1nzy produces almost exactly the same output as the golden.
        Prevents regression.

        TODO(opensource): Get this to run in a reasonable time, it should only take a few seconds, and then create
        the goldens (set REWRITE=True) and then set REWRITE=False.
        I think the issue is in part that the contigmap.length is not working
        '''
        run_inference_test(
            self,
            test_name='test_ame_inference',
            config_name='aa',
            overrides= AME_CASES["M0024_1nzy"],
            rewrite=REWRITE
        )


if __name__ == '__main__':
    unittest.main()
    # test_ame_inference()