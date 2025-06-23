import csv
import glob

joblist_filename = "joblist_dmrg_fss_all_alpha2.5.txt"
output_missing_jobs_filename = "missing_jobs.txt"
results_pattern = "output/spinone_heisenberg_obs_chi500_alpha2.5_L*.csv"

# Step 1: Read job list and extract D values
job_D_values = []

with open(joblist_filename, "r") as f:
    for line in f:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        for i, part in enumerate(parts):
            if part == "-D":
                D_value = float(parts[i + 1])
                break
            elif part.startswith("-D"):
                D_value = float(part[2:])
                break
        else:
            continue  # skip if no -D found
        job_D_values.append((D_value, stripped))

# Step 2: Read all CSV files and extract D values
csv_D_values = set()
for filename in glob.glob(results_pattern):
    with open(filename, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                csv_D_values.add(float(row["D"]))
            except (KeyError, ValueError):
                continue  # Skip malformed or headerless rows


# Step 3: Compare and find missing jobs
missing_jobs = []
for D, job in job_D_values:
    if D not in csv_D_values:
        missing_jobs.append(job)

# Step 4: Write missing jobs to a file
with open(output_missing_jobs_filename, "w") as f:
    for job in missing_jobs:
        f.write(job + "\n")

print(f"Found {len(missing_jobs)} missing jobs. Written to '{output_missing_jobs_filename}'.")

