import subprocess
import psutil
import time
import os

MODELS = ["Informer", "Transformer", "Reformer"]
LOG_DIR = "logs"

TRAIN_ARGS = [
    "--task_name", "long_term_forecast",
    "--is_training", "1",
    "--root_path", "./dataset/electricity/",
    "--data_path", "electricity.csv",
    "--data", "custom",
    "--features", "M",
    "--seq_len", "96",
    "--label_len", "48",
    "--pred_len", "720",
    "--e_layers", "2",
    "--d_layers", "1",
    "--factor", "3",
    "--enc_in", "321",
    "--dec_in", "321",
    "--c_out", "321",
    "--des", "Exp",
    "--itr", "1"
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

    for model in MODELS:
        print(f"========== Running {model} ==========")

        # ログファイル
        gpu_log = os.path.join(LOG_DIR, f"{model}_ECL_96_720_gpu.csv")
        txt_log = os.path.join(LOG_DIR, f"{model}_ECL_96_720.txt")

        # GPU使用量ログ開始
        gpu_proc = start_gpu_logger(gpu_log)
        time.sleep(2)

        try:
            # 学習プロセス実行
            with open(txt_log, "w") as fout:
                subprocess.run(
                    ["python", "-u", "run.py", "--model_id", "ECL_96_720", "--model", model] + TRAIN_ARGS,
                    stdout=fout,
                    stderr=subprocess.STDOUT
                )
        finally:
            stop_gpu_logger(gpu_proc)
            print(f"========== Finished {model} ==========\n")
            time.sleep(3)