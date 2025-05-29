import subprocess
import psutil
import time
import os
from datetime import datetime

LOG_DIR = "logs"
SCRIPTS = [
    "ablation_distilling_gpu_logger.sh",
    "ablation_prob_attention_gpu_logger.sh" # 追加スクリプトも同様に動作
    # さらに追加可能
]

def start_gpu_logger(log_file):
    with open(log_file, "w") as f:
        f.write("timestamp,utilization.gpu [%],memory.used [MiB],memory.total [MiB],processes\n")
    return subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits", "-l", "1"],
        stdout=open(log_file, "a"),
        stderr=subprocess.DEVNULL
    )

def stop_gpu_logger(proc):
    try:
        parent = psutil.Process(proc.pid)
        for child in parent.children(recursive=True):
            child.terminate()
        parent.terminate()
    except psutil.NoSuchProcess:
        pass

if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)

    for script in SCRIPTS:
        base_name = os.path.splitext(os.path.basename(script))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_id = f"{base_name}_{timestamp}"  # 引数として渡すログID

        gpu_log_file = os.path.join(LOG_DIR, f"{log_id}_gpu_log.csv")
        stdout_log_file = os.path.join(LOG_DIR, f"{log_id}_stdout.txt")

        print(f"========== Running {script} with log_id={log_id} ==========")

        gpu_proc = start_gpu_logger(gpu_log_file)
        time.sleep(2)

        try:
            with open(stdout_log_file, "w") as fout:
                # log_id を引数としてスクリプトに渡す
                subprocess.run(["bash", script, log_id], stdout=fout, stderr=subprocess.STDOUT)
        finally:
            stop_gpu_logger(gpu_proc)
            print(f"========== Finished {script} ==========")
            print(f"GPU log saved to: {gpu_log_file}")
            print(f"Stdout log saved to: {stdout_log_file}\n")
            time.sleep(3)  # クールダウン
