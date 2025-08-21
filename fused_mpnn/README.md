# Sequence design example:
```
apptainer exec /software/containers/mlfold.sif python /databases/mpnn/fused_mpnn/run.py \ 
--model_type "ligand_mpnn" \ 
--pdb_path $pdb_path \ 
--out_folder $out_folder_path \ 
--batch_size 1 \
--number_of_batches 1 \
--pack_side_chains 0 #optionally pack side chains
```

# Side chain only packing example:
```
apptainer exec /software/containers/mlfold.sif python /databases/mpnn/fused_mpnn/sc_run.py \ 
--pdb_path $pdb_path \ 
--out_folder $out_folder_path \ 
--pack_side_chains 1 \
--batch_size 1 \
--number_of_batches 1
```

# Design only specified residues:
```
apptainer exec /software/containers/mlfold.sif python /databases/mpnn/fused_mpnn/run.py \
--model_type "ligand_mpnn" \
--pdb_path $pdb_path \
--out_folder $out_folder_path \
--batch_size 1 \
--number_of_batches 1 \
--redesigned_residues "A12 A13 A14 B2 B25" \
--omit_AA "X" \
--pack_side_chains 1 \
--repack_everything 1
```



# Model types:

- protein_mpnn - original ProteinMPNN trained on the whole PDB exluding non-protein atoms
- ligand_mpnn - atomic context aware model trained with small molecules, nucleotides, metals etc on the whole PDB
- per_residue_label_membrane_mpnn - ProteinMPNN model trained with addition label per residue specifying if that residue is buried or exposed
- global_label_membrane_mpnn - ProteinMPNN model trained with global label per PDB id to specify if protein is transmembrane
- soluble_mpnn - ProteinMPNN trained only on soluble PDB ids
- pssm_mpnn - ProteinMPNN with additional PSSM like inputs
- antibody_mpnn - ProteinMPNN trained with bias towards antibody PDBs

