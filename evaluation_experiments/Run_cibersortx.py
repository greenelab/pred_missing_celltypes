
'''
    This runs deconvolution with CIBERSORTx through Docker in the terminal.
    Each user is meant to add their own token, available through the CIBERSORTX team.
    Each user is also meant to add their own email, paths, etc.
    
'''

import subprocess

email_user = "adriana.ivich@cuanschutz.edu"
token_user = "a8df2dc9258bd39ac38c3361e938b2a5"

##################################################################################################################################

in_path = "/Users/ivicha/Documents/Project_missingcelltype/pred_missing_celltypes/data/EXP3/cibersort/"
out_path = "/Users/ivicha/Documents/Project_missingcelltype/pred_missing_celltypes/data/EXP3/cibersort_results/randomprop_nonoise_0missing/"
signal_file = "MCT_adp_EXP3_randomprop_nonoise_0missing_signal.txt"
mixture_file = "MCT_adp_EXP3_randomprop_nonoise_mixture.txt"

docker_command_part1 = f"docker run -v {in_path}:/src/data -v {out_path}:/src/outdir cibersortx/fractions "
docker_command_part2 = f"--username {email_user} --token {token_user} --single_cell TRUE "
docker_command_part3 = f"--refsample {signal_file} --mixture {mixture_file} --fraction 0.75 --perm 500"

# Combine the parts to form the complete Docker command
docker_command = f"{docker_command_part1}{docker_command_part2}{docker_command_part3}"

# Specify the directory where you want to run the Docker command
target_directory = in_path

# Run the Docker command
try:
    subprocess.run(docker_command, check=True, shell=True)
except subprocess.CalledProcessError as e:
    print(f"Error running Docker command: {e}")

##################################################################################################################################

in_path = "/Users/ivicha/Documents/Project_missingcelltype/pred_missing_celltypes/data/EXP3/cibersort/"
out_path = "/Users/ivicha/Documents/Project_missingcelltype/pred_missing_celltypes/data/EXP3/cibersort_results/randomprop_nonoise_2missing/"
signal_file = "MCT_adp_EXP3_randomprop_nonoise_2missing_signal.txt"
mixture_file = "MCT_adp_EXP3_randomprop_nonoise_mixture.txt"

docker_command_part1 = f"docker run -v {in_path}:/src/data -v {out_path}:/src/outdir cibersortx/fractions "
docker_command_part2 = f"--username {email_user} --token {token_user} --single_cell TRUE "
docker_command_part3 = f"--refsample {signal_file} --mixture {mixture_file} --fraction 0.75 --perm 500"

# Combine the parts to form the complete Docker command
docker_command = f"{docker_command_part1}{docker_command_part2}{docker_command_part3}"

# Specify the directory where you want to run the Docker command
target_directory = in_path

# Run the Docker command
try:
    subprocess.run(docker_command, check=True, shell=True)
except subprocess.CalledProcessError as e:
    print(f"Error running Docker command: {e}")

##################################################################################################################################

in_path = "/Users/ivicha/Documents/Project_missingcelltype/pred_missing_celltypes/data/EXP3/cibersort/"
out_path = "/Users/ivicha/Documents/Project_missingcelltype/pred_missing_celltypes/data/EXP3/cibersort_results/randomprop_noise_0missing/"
signal_file = "MCT_adp_EXP3_randomprop_noise_0missing_signal.txt"
mixture_file = "MCT_adp_EXP3_randomprop_noise_mixture.txt"

docker_command_part1 = f"docker run -v {in_path}:/src/data -v {out_path}:/src/outdir cibersortx/fractions "
docker_command_part2 = f"--username {email_user} --token {token_user} --single_cell TRUE "
docker_command_part3 = f"--refsample {signal_file} --mixture {mixture_file} --fraction 0.75 --perm 500"

# Combine the parts to form the complete Docker command
docker_command = f"{docker_command_part1}{docker_command_part2}{docker_command_part3}"

# Specify the directory where you want to run the Docker command
target_directory = in_path

# Run the Docker command
try:
    subprocess.run(docker_command, check=True, shell=True)
except subprocess.CalledProcessError as e:
    print(f"Error running Docker command: {e}")

##################################################################################################################################

in_path = "/Users/ivicha/Documents/Project_missingcelltype/pred_missing_celltypes/data/EXP3/cibersort/"
out_path = "/Users/ivicha/Documents/Project_missingcelltype/pred_missing_celltypes/data/EXP3/cibersort_results/randomprop_noise_2missing/"
signal_file = "MCT_adp_EXP3_randomprop_noise_2missing_signal.txt"
mixture_file = "MCT_adp_EXP3_randomprop_noise_mixture.txt"

docker_command_part1 = f"docker run -v {in_path}:/src/data -v {out_path}:/src/outdir cibersortx/fractions "
docker_command_part2 = f"--username {email_user} --token {token_user} --single_cell TRUE "
docker_command_part3 = f"--refsample {signal_file} --mixture {mixture_file} --fraction 0.75 --perm 500"

# Combine the parts to form the complete Docker command
docker_command = f"{docker_command_part1}{docker_command_part2}{docker_command_part3}"

# Specify the directory where you want to run the Docker command
target_directory = in_path

# Run the Docker command
try:
    subprocess.run(docker_command, check=True, shell=True)
except subprocess.CalledProcessError as e:
    print(f"Error running Docker command: {e}")


##################################################################################################################################

in_path = "/Users/ivicha/Documents/Project_missingcelltype/pred_missing_celltypes/data/EXP3/cibersort/"
out_path = "/Users/ivicha/Documents/Project_missingcelltype/pred_missing_celltypes/data/EXP3/cibersort_results/realisticprop_nonoise_0missing/"
signal_file = "MCT_adp_EXP3_realisticprop_nonoise_0missing_signal.txt"
mixture_file = "MCT_adp_EXP3_realisticprop_nonoise_mixture.txt"

docker_command_part1 = f"docker run -v {in_path}:/src/data -v {out_path}:/src/outdir cibersortx/fractions "
docker_command_part2 = f"--username {email_user} --token {token_user} --single_cell TRUE "
docker_command_part3 = f"--refsample {signal_file} --mixture {mixture_file} --fraction 0.75 --perm 500"

# Combine the parts to form the complete Docker command
docker_command = f"{docker_command_part1}{docker_command_part2}{docker_command_part3}"

# Specify the directory where you want to run the Docker command
target_directory = in_path

# Run the Docker command
try:
    subprocess.run(docker_command, check=True, shell=True)
except subprocess.CalledProcessError as e:
    print(f"Error running Docker command: {e}")

##################################################################################################################################

in_path = "/Users/ivicha/Documents/Project_missingcelltype/pred_missing_celltypes/data/EXP3/cibersort/"
out_path = "/Users/ivicha/Documents/Project_missingcelltype/pred_missing_celltypes/data/EXP3/cibersort_results/realisticprop_nonoise_2missing/"
signal_file = "MCT_adp_EXP3_realisticprop_nonoise_2missing_signal.txt"
mixture_file = "MCT_adp_EXP3_realisticprop_nonoise_mixture.txt"

docker_command_part1 = f"docker run -v {in_path}:/src/data -v {out_path}:/src/outdir cibersortx/fractions "
docker_command_part2 = f"--username {email_user} --token {token_user} --single_cell TRUE "
docker_command_part3 = f"--refsample {signal_file} --mixture {mixture_file} --fraction 0.75 --perm 500"

# Combine the parts to form the complete Docker command
docker_command = f"{docker_command_part1}{docker_command_part2}{docker_command_part3}"

# Specify the directory where you want to run the Docker command
target_directory = in_path

# Run the Docker command
try:
    subprocess.run(docker_command, check=True, shell=True)
except subprocess.CalledProcessError as e:
    print(f"Error running Docker command: {e}")

##################################################################################################################################

in_path = "/Users/ivicha/Documents/Project_missingcelltype/pred_missing_celltypes/data/EXP3/cibersort/"
out_path = "/Users/ivicha/Documents/Project_missingcelltype/pred_missing_celltypes/data/EXP3/cibersort_results/realisticprop_noise_0missing/"
signal_file = "MCT_adp_EXP3_realisticprop_noise_0missing_signal.txt"
mixture_file = "MCT_adp_EXP3_realisticprop_noise_mixture.txt"

docker_command_part1 = f"docker run -v {in_path}:/src/data -v {out_path}:/src/outdir cibersortx/fractions "
docker_command_part2 = f"--username {email_user} --token {token_user} --single_cell TRUE "
docker_command_part3 = f"--refsample {signal_file} --mixture {mixture_file} --fraction 0.75 --perm 500"

# Combine the parts to form the complete Docker command
docker_command = f"{docker_command_part1}{docker_command_part2}{docker_command_part3}"

# Specify the directory where you want to run the Docker command
target_directory = in_path

# Run the Docker command
try:
    subprocess.run(docker_command, check=True, shell=True)
except subprocess.CalledProcessError as e:
    print(f"Error running Docker command: {e}")

##################################################################################################################################

in_path = "/Users/ivicha/Documents/Project_missingcelltype/pred_missing_celltypes/data/EXP3/cibersort/"
out_path = "/Users/ivicha/Documents/Project_missingcelltype/pred_missing_celltypes/data/EXP3/cibersort_results/realisticprop_noise_2missing/"
signal_file = "MCT_adp_EXP3_realisticprop_noise_2missing_signal.txt"
mixture_file = "MCT_adp_EXP3_realisticprop_noise_mixture.txt"

docker_command_part1 = f"docker run -v {in_path}:/src/data -v {out_path}:/src/outdir cibersortx/fractions "
docker_command_part2 = f"--username {email_user} --token {token_user} --single_cell TRUE "
docker_command_part3 = f"--refsample {signal_file} --mixture {mixture_file} --fraction 0.75 --perm 500"

# Combine the parts to form the complete Docker command
docker_command = f"{docker_command_part1}{docker_command_part2}{docker_command_part3}"

# Specify the directory where you want to run the Docker command
target_directory = in_path

# Run the Docker command
try:
    subprocess.run(docker_command, check=True, shell=True)
except subprocess.CalledProcessError as e:
    print(f"Error running Docker command: {e}")

