import argparse
import gc
import time
import torch

from aitemplate.testing.benchmark_pt import benchmark_torch_function # Note: this has a built-in warm up of 5 steps
from pipeline_stable_diffusion_ait import StableDiffusionAITPipeline
from onnxruntime.transformers.benchmark_helper import measure_memory


def create_pipeline(batch_size, force_compile):
    from os.path import exists
    if force_compile or not exists(f"./pipelines/pipeline_with_batch_size_{batch_size}"):
        compile_pipeline(batch_size)


def compile_pipeline(batch_size):
    from os import makedirs
    import subprocess

    makedirs("./pipelines", exist_ok=True)
    compile_log_filename = f"./pipelines/compile_with_batch_size_{batch_size}.log"
    print(f"Begin compiling stable diffusion pipeline with batch size = {batch_size}")
    with open(compile_log_filename, 'w') as compile_log:
        subprocess.run(["python3", "compile.py", "--batch-size", f"{batch_size}"], stdout=compile_log, stderr=subprocess.STDOUT)
    print(f"Finished compiling stable diffusion pipeline with batch size = {batch_size}")


def profile_pipeline(pipe, batch_size):
    prompts = ["a photo of an astronaut riding a horse on mars" for _ in range(batch_size)]
    num_inference_steps = 50
    with torch.autocast("cuda"):
        latency = benchmark_torch_function(num_inference_steps, pipe, prompts, batch_size=batch_size) / 1000
        print(f"Batch size = {batch_size}, latency = {latency} s, throughput = {batch_size / latency} queries/s")

        # Garbage collect before measuring memory
        gc.collect()
        torch.cuda.empty_cache()
        measure_memory(is_gpu=True, func=lambda: pipe(prompts, batch_size=batch_size))


def find_max_batch_size(pipe):
    num_inference_steps = 50
    min_batch_size, max_batch_size = 1, 1024
    while (min_batch_size <= max_batch_size):
        batch_size = min_batch_size + (max_batch_size - min_batch_size) // 2
        print(f"Attempting batch size = {batch_size}")
        try:
            create_pipeline(batch_size)
            prompts = ["a photo of an astronaut riding a horse on mars" for _ in range(batch_size)]
            latency = benchmark_torch_function(num_inference_steps, pipe, prompts, batch_size=batch_size) / 1000
            print(f"Batch size = {batch_size}, latency = {latency} s, throughput = {batch_size / latency} queries/s")

            print(f"Batch size = {batch_size} is too low. Refining search space for min batch size.")
            min_batch_size = batch_size+1
        except:
            print(f"Batch size = {batch_size} is too high. Refining search space for max batch size.")
            max_batch_size = batch_size-1

    print(f"Search is complete. Max batch size = {max_batch_size}.")


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-b',
        '--batch_size',
        required=False,
        type=int,
        default=0,
        help='Batch size needs to be specified so modules are compiled with the right batch size',
    )

    parser.add_argument(
        '-fc',
        '--force_compile',
        required=False,
        action='store_true',
        help='Force compile pipeline for new batch size',
    )
    parser.set_defaults(force_compile=False)

    parser.add_argument(
        '-m',
        '--mode',
        required=True,
        type=str,
        choices=['benchmark', 'search'],
        help='Mode to evaluate pipeline on',
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    create_pipeline(args.batch_size, args.force_compile)

    load_start = time.time()
    pipe = StableDiffusionAITPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=True,
    ).to("cuda")
    load_end = time.time()
    print(f"Model loading took {load_end - load_start} seconds")

    if args.mode == "benchmark":
        assert args.batch_size > 0
        profile_pipeline(pipe, args.batch_size)
    else:
        find_max_batch_size(pipe)


if __name__ == "__main__":
    main()
