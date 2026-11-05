import json

# Fixed settings for the single-PES clean BOMD workflow.
cores = None
num_states = 1
multiplicity = None
start_state = 1
method = None
spin_flip = 0
#basis = "DEF2-TZVP"
#exchange = "WB97X-D"
basis = "6-31+G*"
exchange ="BHHLYP"
gpu = 0
branches = 0
timestep = None
tot_timesteps = None

def load_global_variables():
    global cores, multiplicity, method, timestep, tot_timesteps

    with open("../../inputs.json", "r") as f:
        inputs = json.load(f)

    cores = int(inputs["HPC"]["cores"])
    multiplicity = int(inputs["Molecule_data"]["Multiplicity"])
    method = inputs["Elec_structure"]["method"]
    timestep = float(inputs["Propagation"]["Timestep"])
    tot_timesteps = int(inputs["Propagation"]["Tot_timesteps"])
