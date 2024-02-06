'''
    This bach script is meant to be run in the terminal in the folder it resides in.

    Users are expected to write in the run_id desired, and customize the parameters of the experiment in
    the corresponding file. 

    The files run through this script run multiple notebooks parameterized through papermill.

    The results can be found as notebooks in the results folder of this repository.

'''

# Check if run_id is provided as a command-line argument
if [ -z "$1" ]; then
    echo "Usage: $0 <run_id>"
    exit 1
fi
#And Run
run_id=$1
work_dir=${PWD}
script="${work_dir}/${run_id}.py"
out_dir=${work_dir}/../results/${run_id}/
lsf_file=${out_dir}/${file_id_train}_${file_id_test}_final.lsf

bsub -R "rusage[mem=48GB]" -W 24:00 -n 20 -q "normal" -o ${lsf_file} -J ${run_id} ${script} ${out_dir}
