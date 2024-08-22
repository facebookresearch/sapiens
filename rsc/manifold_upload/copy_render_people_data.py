import argparse
import asyncio
import time

from copy_file import copy
from care.data.io import typed
import subprocess as sp
import tqdm

# export CERT_PATH=/mnt/home/rawalk/my-user-cert.pem
## export CARE_MANIFOLD_CERT_PATH=/mnt/home/rawalk/my-user-cert.pem

async def copy_with_print(src, dst, *args, **kwargs):
    print("src: ", src)
    print("dst: ", dst)
    return await copy(src, dst, *args, **kwargs)


async def gather_with_concurrency(n, *coros):
    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*(sem_coro(c) for c in coros))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="download_hmc_stage_output")
    args = parser.parse_args()

    src = "/mnt/home/rawalk/drive/render_people/"
    tgt = "manifold://codec-avatars-3p-dataset/tree/render_people"

    use_slurm = True

    src_path = src
    tgt_path = tgt

    tasks = []
    tasks.append(
        sp.Popen(
            [
                "srun",
                "-N",
                "1",
                "-n",
                "1",
                "-c",
                "16",
                "--mem",
                "20g",
                "-p",
                "cpuonly",
                "python",
                "copy_file.py",
                src_path,
                tgt_path,
            ]
        )
    )
    time.sleep(0.1)

    print(tasks)

    # if use_slurm:
    #     failed_tasks = []
    #     for task in tqdm.tqdm(tasks):
    #         ret = task.wait()
    #         if ret > 0:
    #             failed_tasks.append(" ".join(task.args))
    #     if len(failed_tasks) > 0:
    #         time.sleep(5)
    #         print(f"Failed {len(failed_tasks)} tasks.")
    #         for task in failed_tasks:
    #             print(task)
    # else:
    #     asyncio.run(gather_with_concurrency(10, *tasks))
