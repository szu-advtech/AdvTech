import subprocess
for i in range(100):
    subprocess.call(["python","calc_metrics_error_detect.py", "ground_truth_manual.csv", "testtest.csv.labeled",str(i/100.0)])
