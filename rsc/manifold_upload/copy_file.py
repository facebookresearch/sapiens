import argparse
import asyncio
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Union, Awaitable, Iterable, TypeVar, Any
import uuid

import aiofiles
import fsspec
import tqdm
from datetime import timedelta

import sys, os

sys.path.insert(0, os.getcwd())

# TODO: switch to use manifold_fsspec module
from care.data.io import file_system  # noqa
from fsspec.asyn import AsyncFileSystem
from fsspec.spec import AbstractFileSystem

_T = TypeVar("_T")

# DEFAULT_JOB_SIZE = 2**28  # 256 MB
DEFAULT_JOB_SIZE = 50 * 2**20
DEFAULT_GROUP_N = 2048 * 4  # 64  # 256MB * 64 = 16GB
PROG_NAME = "Copy File"

logging.basicConfig(
    level=logging.INFO, format="[{name}][{levelname}] {message}", style="{"
)
logger = logging.getLogger(PROG_NAME)


def parse_url(url: str) -> Tuple[AbstractFileSystem, str]:
    # Open FS in asynchronous mode to avoid issue with event loops.
    fs, _, parsed = fsspec.core.get_fs_token_paths(
        url, storage_options={"asynchronous": True}
    )
    return fs, parsed[0]


async def copy_file(
    src: str,
    dst: str,
    sems: Dict[str, asyncio.Semaphore],
    file_size: Optional[int] = None,
    job_size: int = DEFAULT_JOB_SIZE,
) -> Tuple[int, int]:

    src_fs, src_path = parse_url(src)
    dst_fs, dst_path = parse_url(dst)

    # File size will determine whether we need to parallelize the read operation.
    if file_size is None:
        async with sems["ls"]:
            while True:
                try:
                    file_size = await src_fs._size(src_path)
                    break
                except Exception as e:
                    logger.warning(
                        f"Encoutering error ({e}) when listing path {src_path}"
                    )
                    await asyncio.sleep(5)

    # Skip if the file size at destination is already correct. Not robust.
    if dst.startswith("manifold://"):
        while True:
            try:
                dst_file_size = await dst_fs._size(dst_path)
                if dst_file_size == file_size:
                    return 1, file_size
            except FileNotFoundError:
                break
            except Exception:
                logger.warning(f"Encoutering error ({e}) when cating file {src_path}")
                await asyncio.sleep(5)

    else:
        if (await aiofiles.os.path.exists(dst_path)) and (
            file_size == (await aiofiles.os.stat(dst_path)).st_size
        ):
            if file_size < job_size:
                return 1, file_size
            else:

                async def inner(byte_range: Tuple[int, int], sem: asyncio.Semaphore):
                    async with aiofiles.open(dst_path, mode="rb") as f:
                        await f.seek(byte_range[0] + byte_range[1] // 2 - 1024, 0)
                        value = await f.read(1024)
                        # return true if all bytes are \x00, which indicates a failure
                        return all(x == 0 for x in value)

                results = await asyncio.gather(
                    *(
                        inner((start, min(job_size, file_size - start)), sems["read"])
                        for start in range(0, file_size, job_size)
                    )
                )
                if not any(results):
                    return 1, file_size

    if file_size < job_size:
        async with sems["read"]:
            while True:
                try:
                    value = await src_fs._cat_file(src_path)
                    break
                except Exception as e:
                    logger.warning(
                        f"Encoutering error ({e}) when cating file {src_path}"
                    )
                    await asyncio.sleep(5)

        async with sems["write"]:
            while True:
                try:
                    await dst_fs._pipe_file(dst_path, value)
                    break
                except Exception as e:
                    logger.warning(
                        f"Encoutering error ({e}) when checking dst file size {dst_path}"
                    )
                    await asyncio.sleep(5)

    else:
        # assert False
        # TODO: This portion is currently limited to Manifold as source and local as destination.
        # Maybe implement AsyncAbstractBufferedFile for all FSs?

        while True:
            rbar = tqdm.tqdm(
                desc=f"[{PROG_NAME}] Read bytes",
                total=file_size,
                unit_scale=1 / 2**20,
                unit="MB",
                mininterval=0.5,
            )
            wbar = tqdm.tqdm(
                desc=f"[{PROG_NAME}] Write bytes",
                total=file_size,
                unit_scale=1 / 2**20,
                unit="MB",
                mininterval=0.5,
            )

            if dst.startswith("manifold://"):
                staging_path_prefix = f"flat/staging_{str(uuid.uuid4())}"

                async def inner(
                    byte_range: Tuple[int, int], sems: Dict[str, asyncio.Semaphore]
                ):
                    async with sems["read"]:
                        while True:
                            try:
                                value = await src_fs._cat_file(
                                    src_path,
                                    start=byte_range[0],
                                    end=byte_range[0] + byte_range[1],
                                )
                                break
                            except Exception as e:
                                logger.warning(
                                    f"Encoutering error ({e}) when cating file {src_path}"
                                )
                                await asyncio.sleep(5)
                        rbar.update(len(value))
                    async with sems["read"]:
                        stage_file_path = (
                            f"{staging_path_prefix}_{byte_range[0]//job_size}"
                        )
                        while True:
                            try:
                                await dst_fs._pipe_file(stage_file_path, value)
                                break
                            except Exception as e:
                                logger.warning(
                                    f"Encoutering error ({e}) when piping file {stage_file_path}"
                                )
                                await asyncio.sleep(5)
                        wbar.update(len(value))

                await asyncio.gather(
                    *(
                        inner((start, min(job_size, file_size - start)), sems)
                        for start in range(0, file_size, job_size)
                    )
                )
                stage_paths = [
                    f"{staging_path_prefix}_{start//job_size}"
                    for start in range(0, file_size, job_size)
                ]

                async with sems["write"]:
                    while True:
                        try:
                            await dst_fs._manifold_concat(stage_paths, dst_path)
                            break
                        except Exception as e:
                            logger.warning(
                                f"Encoutering error ({e}) when concatenating file {dst_path}"
                            )
                            await asyncio.sleep(5)

                rbar.close()
                wbar.close()

                async with sems["ls"]:
                    try:
                        dst_file_size = await dst_fs._size(dst_path)
                        break
                    except Exception as e:
                        logger.warning(
                            f"Encoutering error ({e}) when checking dst file size {dst_path}"
                        )
                        await asyncio.sleep(5)

                if file_size == dst_file_size:
                    break
                else:
                    logger.error(f"File size mismatched! Path: {dst_path}")

            else:

                async def inner(byte_range: Tuple[int, int], sem: asyncio.Semaphore):
                    async with sem:
                        while True:
                            try:
                                value = await src_fs._cat_file(
                                    src_path,
                                    start=byte_range[0],
                                    end=byte_range[0] + byte_range[1],
                                )
                                break
                            except Exception as e:
                                logger.warning(
                                    f"Encoutering error ({e}) when cating file {src_path}"
                                )
                                await asyncio.sleep(5)
                        rbar.update(len(value))

                    async with aiofiles.open(dst_path, mode="rb+") as f:
                        await f.seek(byte_range[0], 0)

                        await f.write(value)
                        wbar.update(len(value))

                # Create an empty file so that we can later open files in "rb+" mode
                async with aiofiles.open(dst_path, mode="wb") as f:
                    await f.write(b"")
                await asyncio.gather(
                    *(
                        inner((start, min(job_size, file_size - start)), sems["read"])
                        for start in range(0, file_size, job_size)
                    )
                )
                rbar.close()
                wbar.close()

                if file_size == (await aiofiles.os.stat(dst_path)).st_size:
                    break
                else:
                    print(f"File size mismatched! Path: {dst_path}")
    return 1, file_size


async def copy_files(
    jobs: List[Dict[str, Union[str, int]]],
    sems: Dict[str, asyncio.Semaphore],
    job_size: int = DEFAULT_JOB_SIZE,
):
    results = await asyncio.gather(
        *(
            copy_file(job["src"], job["dst"], sems, job["size"], job_size)
            for job in jobs
        )
    )
    file_cnt, byte_cnt = (sum(x) for x in zip(*results))
    return file_cnt, byte_cnt


# Tasks allocated to worker processes through this interface
def sync_copy_files(
    jobs: List[Dict[str, Union[str, int]]],
    max_parallel: Dict[str, int],
    job_size: int = DEFAULT_JOB_SIZE,
    task_id: int = 0,
):
    async def inner():
        sems = {key: asyncio.Semaphore(value) for key, value in max_parallel.items()}
        return await copy_files(jobs, sems, job_size)

    return task_id, asyncio.run(inner())


async def task_dispatcher(
    job_queue: asyncio.Queue,
    max_parallel: Dict[str, int],
    job_size: int = DEFAULT_JOB_SIZE,
    group_n: int = DEFAULT_GROUP_N,
):
    start_time = time.time()
    num_workers = max_parallel.pop("num_workers")
    jobs, tasks = [], []
    task_cnt = [0]

    file_cnt_bar = tqdm.tqdm(
        desc=f"[{PROG_NAME}] Total file count",
        total=0,
        unit_scale=1,
        unit="file",
        mininterval=0.5,
    )
    byte_cnt_bar = tqdm.tqdm(
        desc=f"[{PROG_NAME}] Total byte count",
        total=0,
        unit_scale=1 / 2**20,
        unit="MB",
        mininterval=0.5,
        bar_format="{l_bar}{bar}| {n:.2f}/{total:.2f} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )

    # TODO: record finished/unfished jobs to file if the process encounters issue.

    loop = asyncio.get_running_loop()

    # TODO: kill subprocesses properly
    pool = None
    try:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            jobs_byte_cnt = 0
            jobs_file_cnt = 0

            def jobs_large_enough(file_cnt: int, byte_cnt: int) -> bool:
                return file_cnt >= group_n or byte_cnt >= group_n * job_size

            def submit_jobs(
                jobs: List[Dict[str, Union[str, int]]], file_cnt: int, byte_cnt: int
            ) -> None:
                tasks.append(
                    loop.run_in_executor(
                        pool, sync_copy_files, jobs, max_parallel, job_size, task_cnt[0]
                    )
                )
                logger.info(
                    f"Launching task-{task_cnt[0]} with {file_cnt} files and {byte_cnt/2**30:.2f} GB of data..."
                )
                task_cnt[0] += 1

                # Update progress bars
                file_cnt_bar.total += file_cnt
                file_cnt_bar.refresh()
                byte_cnt_bar.total += byte_cnt
                byte_cnt_bar.refresh()

            def update_progress_bar(file_cnt: int, byte_cnt: int) -> None:
                file_cnt_bar.update(file_cnt)
                byte_cnt_bar.update(byte_cnt)

            while True:
                job = await job_queue.get()
                if job is None:
                    break

                if jobs_large_enough(1, job["size"]):
                    # Avoid dispatching job for one large file with other small files.
                    submit_jobs([job], 1, job["size"])
                else:
                    # Accumulate enough jobs to dispatch to workers
                    jobs.append(job)
                    jobs_file_cnt += 1
                    jobs_byte_cnt += job["size"]

                    if jobs_large_enough(jobs_file_cnt, jobs_byte_cnt):
                        submit_jobs(jobs, jobs_file_cnt, jobs_byte_cnt)
                        jobs = []
                        jobs_byte_cnt = 0
                        jobs_file_cnt = 0

                done_tasks, undone_tasks = [], []
                for task in tasks:
                    if task.done():
                        done_tasks.append(task)
                    else:
                        undone_tasks.append(task)
                tasks = undone_tasks
                for task in done_tasks:
                    task_idx, (file_cnt, byte_cnt) = await task
                    logger.info(
                        f"Finished task-{task_idx} with {file_cnt} files and {byte_cnt/2**30:.2f} GB of data."
                    )
                    update_progress_bar(file_cnt, byte_cnt)

            # Submit all the accumulated jobs.
            if len(jobs) > 0:
                submit_jobs(jobs, jobs_file_cnt, jobs_byte_cnt)

            while len(tasks) > 0:
                done_tasks, tasks = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done_tasks:
                    task_idx, (file_cnt, byte_cnt) = await task
                    logger.info(
                        f"Finished task-{task_idx} with {file_cnt} files and {byte_cnt/2**30:.2f} GB of data."
                    )
                    update_progress_bar(file_cnt, byte_cnt)

    finally:
        if pool is not None and pool._processes is not None:
            # TODO: figure out a way to properly close the pool
            for process in pool._processes.values():
                process.terminate()

    total_time = time.time() - start_time
    file_cnt_bar.close()
    byte_cnt_bar.close()

    # Report stats
    logger.info(
        f"Downloaded {file_cnt_bar.total} files and "
        f"{byte_cnt_bar.total/2**30:.2f} GB of data!"
    )
    logger.info(
        f"All tasks finished in {total_time:.2f}s. "
        f"On average {file_cnt_bar.total/total_time:.2f} files/s and {byte_cnt_bar.total/2**20/total_time:.2f} MB/s."
    )


async def limited_mkdir(fs, path, sems):
    async with sems["exist"]:
        while True:
            try:
                # await fs._makedirs(path, exist_ok=True)
                await fs._mkdir(path, create_parents=False)
                break
            except FileExistsError:
                break
            except Exception as e:
                logger.warning(
                    f"Encoutering error ({e}) when creating directory {path}"
                )
                await asyncio.sleep(5)


async def gather_limited(
    aws: Iterable[Awaitable[_T]], *, concurrency_limit: int
) -> List[Any]:
    """
    Helper to await on a list of Awaitables that limits concurrency, returns
    exceptions along with results, and wraps results or exceptions if desired.

    Usage:
        results = await gather_limited(
            [async_func_1(), ..., async_func_5()], concurrency_limit=2, wrap=True
        )
    """
    sem = asyncio.Semaphore(concurrency_limit)

    async def inner(task):
        async with sem:
            return await task

    results = await asyncio.gather(
        *[inner(awaitable) for awaitable in aws],
        return_exceptions=True,
    )
    return results


async def scan_directory(
    src: str,
    dst: str,
    sems: Dict[str, asyncio.Semaphore],
    job_queue: asyncio.Queue,
) -> None:

    src_fs, src_path = parse_url(src)
    dst_fs, dst_path = parse_url(dst)

    async with sems["ls"]:
        while True:
            try:
                results = await src_fs._ls(src_path, detail=True)
                break
            except Exception as e:
                logger.warning(
                    f"Encoutering error ({e}) when listing directory {src_path}"
                )
                await asyncio.sleep(5)

    # TODO: hack, fix local links
    for res in results:
        if res["type"] == "link":
            print("replacing links")
            res["type"] = "directory" if os.path.isdir(res["destination"]) else "file"

    # TODO: deal with symbolic links.
    unsupported_types = ["link", "other"]
    for res in results:
        if res["type"] in unsupported_types:
            logger.warning(f"Ingoring ({res['type']}): {res['name']}")
    results = [res for res in results if res["type"] not in unsupported_types]

    # Set dst paths.
    for res in results:
        res["dst_name"] = res["name"].replace(src_path, dst_path)

    files = [res for res in results if res["type"] == "file"]
    dirs = [res for res in results if res["type"] == "directory"]

    # # TODO: HACK, Exclude tboard folders
    # non_tboard_dirs = []
    # for res in dirs:
    #     if res["name"].endswith("tboard"):
    #         logger.warning(f"Ingoring tboard directory: {res['name']}")
    #     else:
    #         non_tboard_dirs.append(res)
    # dirs = non_tboard_dirs

    # Create folders
    # await asyncio.gather(
    #     *[dst_fs._makedirs(res["dst_name"], exist_ok=True) for res in dirs],
    # )
    # await asyncio.gather(*[limited_mkdir(dst_fs, res["dst_name"], sems) for res in dirs])

    ######## Uncomment this part
    await gather_limited(
        [limited_mkdir(dst_fs, res["dst_name"], sems) for res in dirs],
        concurrency_limit=1024,
    )
    ######## Uncomment this part

    # Create file copy jobs and scan directories recursively
    # await asyncio.gather(
    #     *[
    #         job_queue.put(
    #             {
    #                 "src": src.replace(src_path, res["name"]),
    #                 "dst": dst.replace(dst_path, res["dst_name"]),
    #                 "size": res["size"],
    #             }
    #         )
    #         for res in files
    #     ],
    #     *[
    #         scan_directory(
    #             src.replace(src_path, res["name"]),
    #             dst.replace(dst_path, res["dst_name"]),
    #             sems,
    #             job_queue,
    #         )
    #         for res in dirs
    #     ],
    # )
    await gather_limited(
        [
            job_queue.put(
                {
                    "src": src.replace(src_path, res["name"]),
                    "dst": dst.replace(dst_path, res["dst_name"]),
                    "size": res["size"],
                }
            )
            for res in files
        ]
        + [
            scan_directory(
                src.replace(src_path, res["name"]),
                dst.replace(dst_path, res["dst_name"]),
                sems,
                job_queue,
            )
            for res in dirs
        ],
        concurrency_limit=128,
    )


async def patched_isdir(fs: AsyncFileSystem, path: str, follow_symlink: bool = True):
    try:
        path_type = (await fs._info(path))["type"]
    except IOError:
        return False

    if follow_symlink:
        return path_type in {"directory", "link"}
    else:
        return path_type == "directory"


async def copy(
    src: str,
    dst: str,
    max_parallel_ls: int = 10,
    max_parallel_read: int = 10,
    num_workers: int = 4,
    job_size: int = DEFAULT_JOB_SIZE,
    group_n: int = DEFAULT_GROUP_N,
) -> None:

    src_fs, src_path = parse_url(src)
    dst_fs, dst_path = parse_url(dst)
    assert isinstance(
        src_fs, AsyncFileSystem
    ), "src filesystem does not support async operations."
    assert isinstance(
        dst_fs, AsyncFileSystem
    ), "dst filesystem does not support async operations."

    # Rate limit the different operations
    max_parallel = {
        "ls": max_parallel_ls,
        "read": max_parallel_read,
        "write": max_parallel_read,
        "num_workers": num_workers,
    }
    sems = {
        "ls": asyncio.Semaphore(max_parallel_ls),
        "read": asyncio.Semaphore(max_parallel_read),
        "write": asyncio.Semaphore(max_parallel_read),
        "exist": asyncio.Semaphore(512),
    }

    if await patched_isdir(src_fs, src_path):
        await limited_mkdir(dst_fs, dst_path, sems)
        # await dst_fs._makedirs(dst_path, exist_ok=True)

        job_queue = asyncio.Queue(maxsize=2**12)  # Max 4k files in the queue.
        loop = asyncio.get_running_loop()
        dispatch_jobs = loop.create_task(
            task_dispatcher(job_queue, max_parallel, job_size, group_n)
        )
        scan_task = loop.create_task(scan_directory(src, dst, sems, job_queue))
        await scan_task
        # Put None object into queue to terminate the job dispatcher
        await job_queue.put(None)
        await dispatch_jobs
    else:
        await limited_mkdir(dst_fs, dst_fs._parent(dst_path), sems)
        # await dst_fs._makedirs(dst_fs._parent(dst_path), exist_ok=True)
        await copy_file(src, dst, sems)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog=PROG_NAME)
    parser.add_argument("src", type=str)
    parser.add_argument("dst", type=str)
    parser.add_argument("-max_parallel_ls", default=8, type=int)
    parser.add_argument("-max_parallel_read", default=128, type=int)
    parser.add_argument("-num_workers", "-j", default=6, type=int)
    args = parser.parse_args()

    asyncio.run(
        copy(
            args.src,
            args.dst,
            args.max_parallel_ls,
            args.max_parallel_read,
            args.num_workers,
        )
    )
