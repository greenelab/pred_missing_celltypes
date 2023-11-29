'''
    This bach script is meant to be run in the terminal in the folder it resides in.

    Users are expected to write in the run_id desired, and customize the parameters of the experiment in
    the corresponding file. 

    The files run through this script run multiple notebooks parameterized through papermill.

    The results can be found as notebooks in the results folder of this repository.

'''

run_id= "EXP1"

work_dir=${PWD}
script= "${work_dir}/${run_id}.py"

out_dir=${work_dir}/../results/EXP1/

lsf_file=${out_dir}/${file_id_train}_${file_id_test}_final.lsf
bsub -R "rusage[mem=48GB]" -W 24:00 -n 20 -q "normal" -o ${lsf_file} -J ${run_id} ${script} ${out_dir} 