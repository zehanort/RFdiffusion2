#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../chai_apptainer/chai.sif" "$0" "$@"'
import os
import sys
from pathlib import Path
import argparse
import ast
import torch
import sys, os, glob
import warnings
import submitit
from typing import Optional, List
from functools import partial
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
import pickle
import gc

from Bio import SeqIO
from Bio.PDB import PDBParser, is_aa
from Bio.SeqUtils import seq1

from dataclasses import dataclass
from typing import Any
import antipickle
from antipickle.adapters import DataclassAdapter
import torch
from chai_typing.typing import Bool, Float, Int
from chai_lab.utils import paths
import torch
from torch import Tensor


class TorchAntipickleAdapter(antipickle.AbstractAdapter):
    typestring = "torch"

    def __init__(self):
        self.cpu_device = torch.device("cpu")

    def check_type(self, obj):
        return type(obj) is torch.Tensor  # ignore inherited classes

    def to_dict(self, obj):
        assert obj.device == self.cpu_device, "serializing only cpu tensors"
        return {"data": antipickle.wrap(obj.numpy())}  # use numpy serialization

    def from_dict(self, d):
        return torch.from_numpy(d["data"])


@dataclass
class ConformerData:
    position: Float[Tensor, "n 3"]
    element: Int[Tensor, "n"]
    charge: Int[Tensor, "n"]
    atom_names: list[str]
    bonds: list[tuple[int, int]]
    symmetries: Int[Tensor, "n n_symm"]

    @property
    def num_atoms(self) -> int:
        num_atoms, _ = self.position.shape
        assert num_atoms == len(self.atom_names)
        return num_atoms

    def gather_atom_positions(
        self, query_atom_names: list[str]
    ) -> tuple[Float[Tensor, "n 3"], Bool[Tensor, "n"]]:
        if self.num_atoms == 0:
            gathered_positions = torch.zeros(len(query_atom_names), 3)
            mask = torch.zeros(len(query_atom_names), dtype=torch.bool)
            return gathered_positions, mask

        atom_indices = {name: i for i, name in enumerate(self.atom_names)}
        indices = torch.tensor(
            [atom_indices.get(name, -1) for name in query_atom_names],
            dtype=torch.int,
        )
        mask = indices != -1
        gathered_positions = self.position[indices] * mask.unsqueeze(-1)

        return gathered_positions, mask


adapters = [TorchAntipickleAdapter(), DataclassAdapter(dict(conf=ConformerData))]

# Append the parent directory of 'chai_lab' to sys.path
chai_lab_path = str(Path(__file__).resolve().parent)
sys.path.insert(0, chai_lab_path)

from chai_lab.chai1 import run_inference

def add_slurm_args(
    parser: Optional[argparse.ArgumentParser] = None,
) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Submits a job to the cluster via submitit"
        )
    parser.add_argument(
        "--slurm_log_path",
        default="slurm_logs",
        type=str,
        help="Path where slurm logs will go. Defaults to `slurm_logs`",
    )
    # parser.add_argument(
    #     "--local",
    #     action="store_true",
    #     help="Set to true to run locally rather than submitting to slurm. This is useful for testing purposes.",
    # )
    parser.add_argument(
        "--slurm_partition",
        type=str,
        default="gpu",
        help="Slurm partition to run job on. Defaults to `gpu`.",
    )
    parser.add_argument(
        "--gpu_type",
        type=str,
        default="a4000",
        help="Which GPUs to run on, SLURM GRES constraint. Defaults to 'a4000'.",
    )
    parser.add_argument(
        "--cpu_memory",
        type=int,
        default=16,
        help="Amount of CPU memory to request for SLURM submission in GB. Defaults to 16.",
    )
    parser.add_argument(
        "--cpus_per_task",
        type=int,
        default=2,
        help="Number of CPU cores to request for SLURM submission. Defaults to 2.",
    )
    parser.add_argument(
        "--timeout_min",
        type=int,
        default=120,
        help="Maximum number of minutes for SLURM job to run. Defaults to 120.",
    )
    parser.add_argument(
        "--max_slurm_jobs_at_once",
        type=int,
        default=16,
        help="Maximum number of array jobs to run at once. Defaults to 16.",
    )
    parser.add_argument(
        "--nodes", type=int, default=1, help="Number of nodes to submit to. Defaults to 1."
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="prediction_job",
        help="A string to indicate what the job should be called when submitting to SLURM.",
    )
    return parser

def create_executor(args: argparse.Namespace) -> submitit.AutoExecutor:
    log_folder = Path(args.slurm_log_path)
    log_folder.mkdir(parents=True, exist_ok=True)

    apptainer_path = Path(__file__).resolve().parent.parent / "chai_apptainer" / "chai.sif"
    print('apptainer_path',apptainer_path)

    executor = submitit.AutoExecutor(folder=log_folder, slurm_python=f"{apptainer_path}")
    executor.update_parameters(
        slurm_partition=args.slurm_partition,
        mem_gb=args.cpu_memory,
        slurm_job_name=args.job_name,
        cpus_per_task=args.cpus_per_task,
        slurm_ntasks_per_node=1,
        slurm_array_parallelism=args.max_slurm_jobs_at_once,
        nodes=args.nodes,
        timeout_min=args.timeout_min,
        slurm_exclude="gpu124,gpu135,gpu111,gpu130,gpu100,gpu109,gpu64,gpu121"
    )
    if args.gpu_type.lower() != "none":
        executor.update_parameters(
            slurm_gres=f"gpu:{args.gpu_type}:1",
        )
        
    if args.gpu_type == " ":
        executor.update_parameters(
            slurm_gres=f"gpu:1",
        )

    return executor

def collect_pdb_files(pdb_input):
    pdb_files = []
    if os.path.isdir(pdb_input):
        pdb_files = glob.glob(os.path.join(pdb_input, '*.pdb'))
    else:
        pdb_files = glob.glob(pdb_input)
    if not pdb_files:
        sys.exit(f"Error: No PDB files found for the input: {pdb_input}")
    return pdb_files

def collect_fasta_files(fasta_input):
    fasta_files = []
    if os.path.isdir(fasta_input):
        fasta_files = glob.glob(os.path.join(fasta_input, '*.fasta')) + glob.glob(os.path.join(fasta_input, '*.fa'))
    else:
        fasta_files = glob.glob(fasta_input)
    if not fasta_files:
        sys.exit(f"Error: No FASTA files found for the input: {fasta_input}")
    return fasta_files

def create_fasta(fasta_path, protein_list=None, ligand_list=None, dna_list=None, rna_seqs=None):
    """Create a FASTA-like file from the provided protein, ligand, and DNA lists."""
    fasta_content = []

    # Add proteins to the FASTA content
    if protein_list:
        for i, protein in enumerate(protein_list):
            fasta_content.append(f">protein|protein-{i+1}")
            fasta_content.append(protein)

    # Add ligands to the FASTA content
    if ligand_list:
        for j, ligand in enumerate(ligand_list):
            fasta_content.append(f">ligand|ligand-{j+1}")
            fasta_content.append(ligand)
    
    # Add DNA sequences to the FASTA content
    if dna_list:
        for k, dna in enumerate(dna_list):
            fasta_content.append(f">dna|dna-{k+1}")
            fasta_content.append(dna)
    
    # Add RNA sequences to the FASTA content
    if rna_seqs:
        for l, rna in enumerate(rna_seqs):
            fasta_content.append(f">rna|rna-{l+1}")
            fasta_content.append(rna)

    # Write the FASTA content to the file
    fasta_path.write_text("\n".join(fasta_content))
    print(f"FASTA file created at: {fasta_path}")
    print("Writing FASTA:\n" + "\n".join(fasta_content))
        
def process_molecule_multiples(molecule_string):
    """Process the string representation of a molecule with a repeat count."""
    protein_parts = molecule_string.split('*')
    if protein_parts[0].isdigit():
        repeat_count = int(protein_parts[0])
        protein_part_list = ast.literal_eval(protein_parts[1])
    else:
        repeat_count = int(protein_parts[1])
        protein_part_list = ast.literal_eval(protein_parts[0])
    
    return repeat_count * protein_part_list

def run_predictions(fasta_dict: dict[Path], output_dir_path: Path, export_jsons: bool, use_esm_embeddings: bool, args):
    """Run inference on the provided FASTA files."""

    chai_lab_path = str(Path(__file__).resolve().parent)
    sys.path.insert(0, chai_lab_path)

    from chai_lab.chai1 import run_inference

    for name, fasta_path in fasta_dict.items():
        if 'cif' in args.structure_output:
            out_path = output_dir_path.joinpath(f"pred.{name}_model_idx_4.cif")
        if 'pdb' in args.structure_output:
            out_path = output_dir_path.joinpath(f"pred.{name}_model_idx_4.pdb")
        if os.path.isfile(out_path):
            print(f"File already exists: {out_path}")
            print("Skipping prediction!")
            continue

        print(f"Running inference on {fasta_path}")
        output_paths = run_inference(
            fasta_file=fasta_path,
            output_dir=output_dir_path,
            name=name,
            num_trunk_recycles=args.num_trunk_recycles, #default: 3
            num_diffn_timesteps=args.num_diffn_timesteps, #default: 200
            seed=args.seed, #default: 42
            device=torch.device("cuda:0"),
            use_esm_embeddings=use_esm_embeddings,
            export_jsons=export_jsons,
            args=args
        )
        print(f"Inference with seed {args.seed} completed for {fasta_path}. Output files are stored in: {output_paths}")

        # Memory cleanup
        del output_paths
        torch.cuda.empty_cache()
        gc.collect()

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    #chunk a dictionary into n chunks
    for i in range(0, len(lst), n):
        yield dict(list(lst.items())[i:i + n])

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Predict folder of PDBs or FASTAs with chai-1.')
    parser.add_argument('--pdb_folder', type=str, help='Folder with PDBs to score', required=False)
    parser.add_argument('--pdb_paths_file', type=str, help='File with newline separated PDBs to score', required=False)
    parser.add_argument('--fasta_folder', type=str, help='Folder with FASTA files to score', required=False)
    parser.add_argument('--name', type=str, help='Name of prediction', required=False)
    parser.add_argument('--protein', type=str, help='List of proteins in format "[\'AAAA\',\'GGGG\']"', required=False)
    parser.add_argument('--dna', type=str, help='List of DNA in format "[\'ATGTC\',\'ATGTC\']"', required=False)
    parser.add_argument('--rna', type=str, help='List of RNA in format "[\'AUGUC\',\'AUGUC\']"', required=False)
    parser.add_argument('--ligand', type=str, help='List of ligands in format "[\'CCCCCC(=O)O\',\'CCCCCC(=O)O\']"', required=False)
    parser.add_argument('--output_dir', type=str, help='Directory to store output files', default='./tmp/outputs')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--submit', action='store_true', help='Submit jobs using SLURM array jobs')
    parser.add_argument('--chunk_size', type=int, default=1, help='Number of FASTA files per job in the array job')
    parser.add_argument('--export_mode', type=str, help='Choose metric export mode; json and/or npz',nargs='*', choices=['json', 'npz'], default=['json'])
    parser.add_argument('--structure_output', type=str, help='Choose if structure should be output as cif or pdb',nargs='*', choices=['cif', 'pdb'], default=['pdb'])
    parser.add_argument('--export_arrays', action='store_true', help='Export full plddt and pae arrays')
    parser.add_argument('--seed', type=int, help='Set random seed for chai', default=42)
    parser.add_argument('--export_seed', action='store_true', help='Export the seed used for prediction')
    parser.add_argument('--save_latents', action='store_true', help='Write latents')
    parser.add_argument('--setup', action='store_true', help='setup for batch submission')
    parser.add_argument('--override_ccd', type=str, nargs="+", help="Ligand name3's found in input structures that should not be parsed as CCD ligands", default='')
    parser.add_argument('--num_trunk_recycles', type=int, help='Number of trunk recycles', default=3)
    parser.add_argument('--num_diffn_timesteps', type=int, help='Number of diffn timesteps', default=200)
    parser.add_argument('--log_dir', type=str, help='Directory to store log files', default='./tmp/outputs/chai_logs')
    parser.add_argument('--allow_ccd_pdb_mismatch', action='store_true', help='Predict the structure using the ligand from the CCD even if it has a different number of atoms than your input PDB')
    parser.add_argument('--omit_esm_embeddings', action='store_true')
    
    # Add SLURM-specific arguments
    parser = add_slurm_args(parser)
    defaults = parser.parse_args([])  # Parse with no arguments to get defaults
    args = parser.parse_args()
    
    current_folder = os.getcwd()

    data_to_save = {
        'args': args,
        'current_folder': current_folder,
        'chai_lab_path': chai_lab_path
    }

    args_out = args.log_dir
    os.makedirs(args_out, exist_ok=True)

    args_filename = f"args_{timestamp}.pkl"
    args_file_path = os.path.join(args_out, args_filename)
    
    try:
        with open(args_file_path, 'wb') as file:
            pickle.dump(data_to_save, file)
    except:
        pass
    
    # if (args.protein or args.ligand) and (args.fasta_folder or args.pdb_folder):
    #     parser.error('Please decide between providing a list of proteins/ligands or a folder with FASTA files to score')

    if args.output_dir:
        output_dir_path = Path(args.output_dir)
    elif args.pdb_folder:
        output_dir_path = Path(args.pdb_folder)
    elif args.fasta_folder:
        output_dir_path = Path(args.fasta_folder)
    else:
        output_dir_path = Path('./tmp/outputs')

    output_dir_path.mkdir(parents=True, exist_ok=True)  # Create the output directory if it doesn't exist

    # Initialize fasta_dict
    fasta_dict = {}

    print(f"Output directory: {output_dir_path}")
    
    if args.protein:
        if '*' in args.protein:
            protein_list = process_molecule_multiples(args.protein)
        else:
            protein_list = ast.literal_eval(args.protein)
    else:
        protein_list = []

    if args.ligand:
        ligand_list = ast.literal_eval(args.ligand)
    else:
        ligand_list = []

    if args.dna:
        dna_list = ast.literal_eval(args.dna)
    else:
        dna_list = []

    if args.rna:
        rna_list = ast.literal_eval(args.rna)
    else:
        rna_list = []

    import antipickle
    chai_ccd = antipickle.load(
        paths.cached_conformers.get_path(),
        adapters=adapters
    )
    
    ccd_id_path = Path(f'{chai_lab_path}/ccd_id.txt')
    if not ccd_id_path.is_file():
        raise FileNotFoundError(f"ccd_id.txt not found at {ccd_id_path}")
    ccd_id = [line.rstrip('\n') for line in open(ccd_id_path)]

    if args.pdb_folder or args.pdb_paths_file:

        if args.pdb_folder:
            print(f"Processing PDB files in {args.pdb_folder}")
            PDBFiles = collect_pdb_files(args.pdb_folder)
            if not PDBFiles:
                warnings.warn(f"No PDB files found for the input: {args.pdb_folder}")
        elif args.pdb_paths_file:
            print(f"Processing PDB filenames listed in {args.pdb_paths_file}")
            PDBFiles = []
            with open(args.pdb_paths_file) as f:
                PDBFiles = [l.strip() for l in f.readlines()]
        else:
            raise Exception("should not happen")

        nucleotide_mapping = {
            'DA': 'A',
            'DG': 'G', 
            'DC': 'C', 
            'DT': 'T'}

        rna_mapping = {
            'A': 'A',
            'U': 'U',
            'G': 'G',
            'C': 'C'
        }

        for PDBFile in PDBFiles:
            print(f"Processing PDB file: {PDBFile}")

            filename = os.path.basename(PDBFile)[:-4]
            parser_pdb = PDBParser(QUIET=True)
            structure = parser_pdb.get_structure('structure', PDBFile)

            protein_seqs = []
            dna_seqs = []  # List to store DNA sequences
            rna_seqs = []  # List to store RNA sequences

            for model in structure:
                for chain in model:
                    ccd_ligand_list = [] # List to store ligands listed in the CCD
                    protein_seq = ""
                    dna_seq = ""
                    rna_seq = ""
                    for residue in chain:
                        resname = residue.resname.strip()
                        if resname == 'HOH':  # Skip waters
                            continue
                        if is_aa(resname, standard=True):
                            protein_seq += seq1(resname)
                        elif resname in nucleotide_mapping:  # DNA nucleotides
                            dna_seq += nucleotide_mapping[resname]
                        elif resname in rna_mapping:  # RNA nucleotides
                            rna_seq += rna_mapping[resname]
                        elif resname in chai_ccd.keys():
                            if resname in args.override_ccd:
                                print(f"Skipping parsing {resname} as CCD residue")
                                continue
                            if not args.allow_ccd_pdb_mismatch:
                                if len(residue.child_list) != len(chai_ccd[resname].atom_names):
                                    print(f"{resname} is expected to be in this format:{chai_ccd[resname].atom_names}")
                                    assert False, f"The number of atoms in conformers data for residue {resname} does not match the number of atoms in your ligand in the pdb you provided."

                                for atom_id, atom in enumerate(residue):
                                    if chai_ccd[resname].atom_names[atom_id] == atom.name:
                                        pass
                                    else:
                                        print(f"{resname} is expected to be in this format:{chai_ccd[resname].atom_names}")
                                        assert False, f"Atom names and/or order in conformers data for residue {resname} does not match the ligand in your pdb."

                            warnings.warn("This is only valid if you stick to standard CCD ligand namings. Otherwise please use smiles ligands.")
                            ccd_ligand_list.append(f'({resname})')
                        else:
                            assert False, f"Residue - {resname} - not in CCD. Please add it as a smiles ligand."
                    
                    # Append sequences if they are not empty
                    if protein_seq:
                        protein_seqs.append(protein_seq)
                    protein_seqs.extend(ccd_ligand_list)
                    if dna_seq:
                        dna_seqs.append(dna_seq)
                    if rna_seq:
                        rna_seqs.append(rna_seq)

            # Add the ligand sequences to ligand_seqs if applicable
            ligand_seqs = ligand_list

            # Extend the protein sequences with additional protein list if needed
            protein_seqs.extend(protein_list)
            dna_seqs.extend(dna_list)
            rna_seqs.extend(rna_list)

            # Path to the output FASTA file
            fasta_filename = f"{filename}.fasta"
            fasta_path = Path(f"{output_dir_path}/fastas/{fasta_filename}")

            # Create the output directory if it doesn't exist
            os.makedirs(f"{output_dir_path}/fastas", exist_ok=True)

            # Create the FASTA file with all sequences
            create_fasta(fasta_path, protein_seqs, ligand_seqs, dna_seqs, rna_seqs)

            # Add the fasta_path to fasta_dict 
            fasta_dict[filename] = fasta_path
                
    if args.fasta_folder:
        print(f"Processing FASTA inputs from: {args.fasta_folder}")
        fasta_files = collect_fasta_files(args.fasta_folder)

        if args.protein or args.ligand:
            sys.exit(f"You provided a list of proteins/ligands/DNA/RNA in addition to folder with FASTA files to score. The provided list would be ignored. Stopping for now.")

        for fasta_file in fasta_files:
            filename = os.path.basename(fasta_file)
            if filename.endswith('.fasta'):
                filename = filename[:-6]
            elif filename.endswith('.fa'):
                filename = filename[:-3]
            else:
                warnings.warn(f"Unexpected FASTA file extension for {fasta_file}. Proceeding anyway.")

            fasta_dict[filename] = Path(fasta_file)

    if (protein_list or ligand_list) and not (args.fasta_folder or args.pdb_folder):
        # Path to the output FASTA file
        fasta_path = Path(f"{output_dir_path}/{args.name}.fasta")

        # Create the FASTA file
        create_fasta(fasta_path, protein_list, ligand_list, dna_list, rna_list)
        fasta_dict[args.name] = Path(fasta_path)
        
    def get_non_default_args(args, defaults):
        """
        Compare parsed args with defaults and return a dictionary of arguments that differ from defaults.

        Args:
            args (Namespace): Parsed command-line arguments.
            defaults (Namespace): Default values of command-line arguments.

        Returns:
            dict: Arguments that have been set by the user (non-default).
        """
        non_default = {}
        for key, value in vars(args).items():
            default_value = getattr(defaults, key, None)
            if value != default_value:
                non_default[key] = value
        return non_default

    if args.setup:
        non_default_args = get_non_default_args(args, defaults)

        exclude_args = {'fasta_folder', 'output_dir', 'setup', 'pdb_list'}
        filtered_args = {k: v for k, v in non_default_args.items() if k not in exclude_args}

        arg_strings = []
        for arg, value in filtered_args.items():
            arg_name = f"--{arg}"
            if arg in ["slurm_partition", "gpu_type", "chunk_size", "ligand", "pdb_folder", "dna", "rna"]:
                continue
            if isinstance(value, bool):
                if value:
                    arg_strings.append(arg_name)
            elif isinstance(value, list):
                for item in value:
                    arg_strings.append(f"{arg_name} {item}")
            else:
                arg_strings.append(f"{arg_name} {value}")

        additional_args = ' '.join(arg_strings)
        
        with open(f"./cmds", 'w') as f:
            for name, fasta_path in fasta_dict.items():
                cmd = f"/net/software/lab/chai/chai-lab/run_chai.sh --fasta_folder {fasta_path} --output_dir {output_dir_path} {additional_args}\n"
                f.write(cmd)
        return


    if args.submit:
        chunked_fasta_dict = list(chunks(fasta_dict, args.chunk_size))
        
        # Create the executor with SLURM parameters
        executor = create_executor(args)
        
        jobs = []
        for chunk in chunked_fasta_dict:
            # Create a partial function with the necessary arguments
            run_fn = partial(run_predictions, fasta_dict=chunk, output_dir_path=output_dir_path, export_jsons=args.debug, args=args)
            job = executor.submit(run_fn)
            jobs.append(job)
        
        print(f"Submitted {len(jobs)} jobs.")
    else:
        run_predictions(fasta_dict, output_dir_path, args.debug, not args.omit_esm_embeddings, args)
    
if __name__ == "__main__":
    main()
