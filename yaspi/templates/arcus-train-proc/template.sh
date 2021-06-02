#!/bin/bash
#SBATCH --job-name={{job_name}}
#SBATCH --array={{array}}
#SBATCH --time={{time_limit}}
#SBATCH --output={{log_path}}
#SBATCH --partition={{partition}}
#SBATCH --ntasks-per-node={{ntasks_per_node}}
#SBATCH --gres gpu:1
#SBATCH --constraint='gpu_gen:Kepler'
#SBATCH --account=phys-qsum
#SBATCH --qos=priority
{{sbatch_resources}}
{{exclude_nodes}}
# -------------------------------

# enable terminal stdout logging
echo "linking job logs to terminal"
echo "=================================================================="

{{env_setup}}

# Run the loop of runs for this task.
worker_id=$((SLURM_ARRAY_TASK_ID - 1))
echo "This is SLURM task $SLURM_ARRAY_TASK_ID, worker id $worker_id"
declare -a custom_args_queue=({{job_queue}})

# handle potential ipython issues with history
export IPYTHONDIR=/tmp

prep="{{prep}}"
echo "running prep cmd $prep"
eval "${prep}" 

cmd="{{cmd}}"
cmd="srun --unbuffered ${cmd} ${custom_args_queue[${worker_id}]}"
echo "running cmd $cmd"
eval "${cmd}" 