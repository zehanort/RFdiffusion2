import logging
import os
from pathlib import Path
import subprocess as sp
import time
from typing import Union

from fastapi import FastAPI, File, Form, UploadFile, HTTPException


BASE_DIR = Path("/outputs")
MODEL_WEIGHTS_PT = Path("/opt/RFdiffusion2/rf_diffusion/model_weights/RFD_173.pt")

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="RFdiffusion 2 API")


@app.get("/v1/health/ready")
async def health_check():
    return {"status": "ready"}


@app.post("/api/generate")
async def generate(
    pdb_file: UploadFile = File(...),
    contigmap: str = Form(""),
    num_designs: int = Form(10),
    ligand_id: str = Form(None),
    diffuser_T: int = Form(50),
    diffuser_partial_T: Union[int, float] = Form(None),
    run_output_dirname: str = Form(""),
    sanitize_output_filenames: bool = Form(True)
):
    # read the pdb file, if it exists
    if not pdb_file.filename.endswith(".pdb"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only .pdb files are accepted.")
    if not pdb_file:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    if run_output_dirname == "":
        raise HTTPException(status_code=400, detail="The run output directory must be specified.")

    # make the output directory if it doesn't exist
    output_dir = BASE_DIR / run_output_dirname
    log_inputs_dir = output_dir / "inputs"
    log_inputs_dir.mkdir(parents=True, exist_ok=True)

    # save the uploaded input file
    logged_input_pdb = log_inputs_dir / os.path.basename(pdb_file.filename)
    with open(logged_input_pdb, "wb") as f:
        f.write(pdb_file.file.read())
        f.flush()

    # build the command
    if not contigmap.startswith("[") and not contigmap.endswith("]"):
        contigmap = ''.join(contigmap.split())
        contigmap = f'["{contigmap}"]'
    rfdiffusion_cmd = [
        "python",
        "run_inference.py",
        f"inference.input_pdb={logged_input_pdb}",
        f"inference.output_prefix={output_dir / 'design'}",
        f"inference.ckpt_path={MODEL_WEIGHTS_PT}",
        f"inference.num_designs={num_designs}",
        f"contigmap.contigs={contigmap}",
        f"diffuser.T={diffuser_T}",
    ]
    if ligand_id is not None:
        rfdiffusion_cmd.append(f"inference.ligand={ligand_id}")
    if diffuser_partial_T is not None:
        rfdiffusion_cmd.append(f"diffuser.partial_T={diffuser_partial_T}")

    # run the command
    start_time = time.time()
    try:
        logging.info(f"Running RFdiffusion2 with command: {' '.join(rfdiffusion_cmd)}")
        _ = sp.run(rfdiffusion_cmd, check=True)
    except sp.CalledProcessError as e:
        logging.error(f"Command failed with error: {e.stderr}")
        raise HTTPException(status_code=500, detail="Command failed")
    elapsed_time = time.time() - start_time

    # post-processing output filenames
    if sanitize_output_filenames:
        targets = [
            (output_dir, ["pdb", "trb"]),
            (output_dir / "unidealized", ["pdb"]),
        ]
        for folder, extensions in targets:
            for ext in extensions:
                for outfile in folder.glob(f"design_*.{ext}"):
                    new_filepath = outfile.with_name(f"{outfile.stem.split('-')[0]}.{ext}")
                    outfile.rename(new_filepath)

    return {
        "status": "success",
        "output_dir": str(output_dir),
        "elapsed_time": elapsed_time,
    }
