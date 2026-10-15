#!/usr/bin/env python3
"""Wait for one identified predecessor process and then for a free physical GPU."""

import argparse
from pathlib import Path
import subprocess
import time


def process_start_ticks(pid, proc_root=Path("/proc")):
    try:
        raw = (proc_root / str(pid) / "stat").read_text()
    except FileNotFoundError:
        return None
    # Fields following the final ')' start with field 3; starttime is field 22.
    return raw.rsplit(")", 1)[1].split()[19]


def parse_gpu_processes(output, gpu_uuid):
    return [
        fields[1].strip()
        for line in output.splitlines()
        if len(fields := line.split(",")) == 2 and fields[0].strip() == gpu_uuid
    ]


def wait_for_gpu(gpu, pid, start_ticks, status_file, poll_seconds=15):
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    status_file = Path(status_file)
    status_file.write_text(f"waiting: GPU {gpu} predecessor PID {pid}\n")
    print(f"Waiting for predecessor PID {pid}, start ticks {start_ticks}", flush=True)
    while process_start_ticks(pid) == str(start_ticks):
        time.sleep(poll_seconds)
    print("Predecessor exited; checking physical GPU availability", flush=True)
    gpu_uuid = subprocess.check_output(
        ["nvidia-smi", f"--id={gpu}", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    if not gpu_uuid.startswith("GPU-") or "\n" in gpu_uuid:
        raise RuntimeError(f"Could not identify GPU {gpu}: {gpu_uuid!r}")
    previous = None
    while True:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader",
            ],
            text=True,
        )
        busy = parse_gpu_processes(output, gpu_uuid)
        if not busy:
            break
        status_file.write_text(
            f"waiting: GPU {gpu} compute processes {','.join(busy)}\n"
        )
        if busy != previous:
            print(f"GPU {gpu} still occupied by PIDs {busy}", flush=True)
            previous = busy
        time.sleep(poll_seconds)
    status_file.write_text(f"ready: GPU {gpu}\n")
    print(f"Physical GPU {gpu} is free", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--start-ticks", required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    args = parser.parse_args()
    wait_for_gpu(args.gpu, args.pid, args.start_ticks, args.status_file)


if __name__ == "__main__":
    main()
