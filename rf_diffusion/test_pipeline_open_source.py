import os
import glob
import shutil
import assertpy
import subprocess
import unittest

import pandas as pd
from icecream import ic
import pytest

import error
import benchmark.sweep_hyperparam
ic.configureOutput(includeContext=True)

class TestBenchmark(unittest.TestCase):

    def test_arg_combos(self):
        for arg_str, want in [
            ('''
        a=1|2|3
        ''',
        [
            {'a':'1'},
            {'a':'2'},
            {'a':'3'}
        ]
        ),
        ('''
        (a=1 b=2)|(a=3 b=4)
        ''',
        [
            {'a':'1', 'b':'2'},
            {'a':'3', 'b':'4'},
        ]),
        ('''
        c=5
        (a=1 b=2)|(a=3 b=4)
        ''',
        [
            {'c':'5','a':'1', 'b':'2'},
            {'c':'5','a':'3', 'b':'4'},
        ]),

        ('''
        a=1
        (b=2)|(a=3 b=4)
        ''',
        [
            {'a':'1', 'b':'2'},
            {'a':'3', 'b':'4'},
        ]),
        ('''
        a=1
        (a=2 b=3)|(b=4|5)
        ''',
        [
            {'a':'2', 'b':'3'},
            {'a':'1', 'b':'4'},
            {'a':'1', 'b':'5'},
        ]),
        ('''
        (a=1)|(b=4|5)
        ''',
        [
            {'a':'1'},
            {'b':'4'},
            {'b':'5'},
        ]),
        ('''
        (a=1)|((b=4)|(b=5))
        ''',
        [
            {'a':'1'},
            {'b':'4'},
            {'b':'5'},
        ]),
        ('''
        a=1|
            2|
            3
        ''',
        [
            {'a':'1'},
            {'a':'2'},
            {'a':'3'},
        ]),
        ('''
        a=1
        ()|(a=2)
        ''',
        [
            {'a':'1'},
            {'a':'2'},
        ]),
        ('''
        ()|(a=2)
        a=1
        ''',
        [
            {'a':'1'},
            {'a':'1'},
        ]),
        ('''
        POST(()|(a=2))
        a=1
        ''',
        [
            {'a':'1'},
            {'a':'2'},
        ]),
        ('''
        a=1
        (b=3)|(a=2)
        ''',
        [
            {'a':'1', 'b':'3'},
            {'a':'2'},
        ]),
        ('''
        a=1
        (b=3)|(a=2|3)
        ''',
        [
            {'a':'1', 'b':'3'},
            {'a':'2'},
            {'a':'3'},
        ]),
        # TODO: Implement check such that this testcase returns an error, as it is somewhat nonsensical.
        # ('''
        # arg1=A|B
        # (arg1=C arg3=D)|(arg1=c arg3=d)
        # ''',
        # ['error']
        # ),
        ('''
        a=1
        b=2
        *benchmark/test_benchmarks.txt
        ''',
        [
            {'a':'1', 'b':'2', 'c':'3'},
            {'a':'1', 'b':'2', 'c':'4'},
        ]),
        ('''
        a=(ddd eee fff)
        b=2
        ''',
        [
            {'a':'ddd eee fff', 'b':'2'}
        ]),
        ]:
            with error.context(f'{arg_str=} {want=}'):
                got = benchmark.sweep_hyperparam.parse_arg_str(arg_str)
                ic(got, want)
                self.assertEqual(got, want)

    def test_subprocess_retcode(self):
        job = '''python raise_exception.py'''
        print(f'running job: {job}')
        proc = subprocess.run(job, shell=True)
        print(f'{proc=}') 
        print(f'{proc.returncode=}')
        assertpy.assert_that(proc.returncode).is_not_equal_to(0)
    
    def test_pipeline_completes(self):
        '''
        Tests that the pipeline runs end-to-end and produces the appropriate metrics
        for a toy input.
        '''

        expected_number_of_sequences = 2
        outdir = os.path.abspath('test_outputs/pipeline_0')
        if os.path.exists(outdir):
            shutil.rmtree(outdir)

        # The following commented out caching line breaks the test currently due to MPNN nondeterminism.
        # # Speeds up the process by making the expected AF2 outputs, so that AF2 doesn't have to run.
        # shutil.copytree('test_data/pipeline_0', outdir)

        job = f'''./benchmark/pipeline.py --config-name=pipeline_test in_proc=1 outdir={outdir}'''
        print(f'running job: {job}')
        proc = subprocess.run(job, shell=True)
        print(f'{proc=}')
        print(f'{proc.returncode=}')
        assertpy.assert_that(proc.returncode).is_equal_to(0)

        expected_metrics_csv_path = os.path.join(outdir, 'compiled_metrics.csv')
        assert os.path.exists(expected_metrics_csv_path)

        df = pd.read_csv(expected_metrics_csv_path)
        assertpy.assert_that(df.shape[0]).is_equal_to(expected_number_of_sequences)

        success_metric = 'backbone_aligned_allatom_rmsd_af2_unideal_sym_resolved'
        assert df[success_metric].notna().all(), f'expected non nans: {df[success_metric]=}'

    @pytest.mark.open_source_gpu
    def test_chai_pipeline_completes(self):
        '''
        Tests that the pipeline runs end-to-end with:
            - a single 30-diffusion-step design run with RFdiffusion2
            - a single sequence per structure fit with LigandMPNN
            - Chai for the final evaluation
            - Motif self-consistency metrics are computed
         
        Note: This test is expected to run on a GPU, as it uses Chai and 3 diffusion steps
        of a 180-residue protein, i.e. run with:
        apptainer exec --nv exec/bakerlab_rf_diffusion_aa.sif pytest --disable-warnings -s --full-trace test_pipeline_open_source.py::TestBenchmark::test_chai_pipeline_completes
        It ought to take 4-8 minutes to run on a single GPU.

        # TODO(opensouce): Add a longer (~30 timesteps version of this test) with some more reasonable assertions here, i.e. check that the 
        # output PDB is reasonably self-consistent.
        '''
        expected_number_of_sequences = 1
        outdir = os.path.abspath('test_outputs/pipeline_chai')

        job = f'''./benchmark/pipeline.py --config-name=enzyme_bench_n41_e2e_test outdir={outdir}'''
        
        if os.path.exists(outdir):
            shutil.rmtree(outdir)

        print(f'running job: {job}')
        proc = subprocess.run(job, shell=True)
        print(f'{proc=}')
        print(f'{proc.returncode=}')
        assertpy.assert_that(proc.returncode).is_equal_to(0)

        expected_metrics_csv_path = os.path.join(outdir, 'compiled_metrics.csv')
        assert os.path.exists(expected_metrics_csv_path)

        df = pd.read_csv(expected_metrics_csv_path)
        assertpy.assert_that(df.shape[0]).is_equal_to(expected_number_of_sequences)

        success_metric = 'backbone_aligned_allatom_rmsd_chai_unideal_all_chaimodel_0_sym_resolved'
        if success_metric not in df.columns:
            raise ValueError(f'Expected metric {success_metric} not found in {df.columns=}')
        assert df[success_metric].notna().all(), f'expected non nans: {df[success_metric]=}'
    
    def test_chai_step(self):
        '''
        Tests that the Chai step of the pipeline runs.
        '''
        outdir = os.path.abspath('test_outputs/chai_step')
        if os.path.exists(outdir):
            shutil.rmtree(outdir)
        shutil.copytree('test_start_points/chai_step', outdir)

        job = f'''./benchmark/pipeline.py --config-name=enzyme_bench_n41_e2e_test outdir={outdir} start_step=score stop_step=score score.chai1.omit_esm_embeddings=True'''
        print(f'running job: {job}')
        proc = subprocess.run(job, shell=True)
        print(f'{proc=}')
        print(f'{proc.returncode=}')
        assertpy.assert_that(proc.returncode).is_equal_to(0)
    
    def test_without_chai_step(self):
        '''
        Test the pipeline runs end-to-end while spoofing the Chai step.
        This is useful for smoke-testing the pipeline on machines that
        do not meet Chai's GPU requirements (namely bfloat16 support).
        '''

        expected_number_of_sequences = 1
        outdir = os.path.abspath('test_outputs/pipeline_without_chai_step')        
        if os.path.exists(outdir):
            shutil.rmtree(outdir)

        # Run the pipeline up to the chai step
        job = f'''./benchmark/pipeline.py --config-name=enzyme_bench_n41_e2e_test stop_step=thread_mpnn outdir={outdir}'''
        proc = subprocess.run(job, shell=True)
        assertpy.assert_that(proc.returncode).is_equal_to(0)

        # Spoof chai outputs
        mpnn_pdbs = glob.glob(os.path.join(outdir, 'ligmpnn/packed/*.pdb'))
        assert len(mpnn_pdbs) == 1, f'Expected 1 MPNN PDB, got {len(mpnn_pdbs)}: {mpnn_pdbs}'
        mpnn_pdb = mpnn_pdbs[0]
        head, tail = os.path.split(mpnn_pdb)
        tail = tail.split('.')[0]
        for model_idx in range(5):
            chai_path = os.path.join(head, 'chai1/out', f'pred.{tail}_model_idx_{model_idx}.pdb')
            os.makedirs(os.path.dirname(chai_path), exist_ok=True)
            shutil.copy(mpnn_pdb, chai_path)
        # Copy some dummy scores
        for score_json in glob.glob('test_data/pipeline_without_chai_step/chai_scores/*'):
            tail = os.path.basename(score_json)
            shutil.copy(score_json, os.path.join(outdir, 'ligmpnn/packed/chai1/out', tail))

        job = f'''./benchmark/pipeline.py --config-name=enzyme_bench_n41_e2e_test start_step=metrics outdir={outdir}'''
        proc = subprocess.run(job, shell=True)
        assertpy.assert_that(proc.returncode).is_equal_to(0)

        # Run the pipeline following the chai step
        expected_metrics_csv_path = os.path.join(outdir, 'compiled_metrics.csv')
        assert os.path.exists(expected_metrics_csv_path)

        df = pd.read_csv(expected_metrics_csv_path)
        assertpy.assert_that(df.shape[0]).is_equal_to(expected_number_of_sequences)

        success_metric = 'backbone_aligned_allatom_rmsd_chai_unideal_all_chaimodel_0_sym_resolved'
        if success_metric not in df.columns:
            raise ValueError(f'Expected metric {success_metric} not found in {df.columns=}')
        assert df[success_metric].notna().all(), f'expected non nans: {df[success_metric]=}'

if __name__ == '__main__':
        unittest.main()
