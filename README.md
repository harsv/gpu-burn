Update 30-11-2016: Versions 0.7 and up also benchmark.

I work with GPUs a lot and have seen them fail in a variety of ways: too much (factory) overclocked memory/cores, unstable when hot, unstable when cold (not kidding), memory partially unreliable, and so on. What's more, failing GPUs often fail silently and produce incorrect results when they are just a little unstable, and I have seen such GPUs consistently producing correct results on some apps and incorrect results on others.

What I needed in my tool box was a stress test for multi-GPGPU-setups that used all of the GPUs' memory and checked the results while keeping the GPUs burning. There are not a lot of tools that can do this, let alone for Linux. Therefore I hacked together my own. It runs on Linux and uses the CUDA driver API.

My program forks one process for each GPU on the machine, one process for keeping track of the GPU temperatures if available (e.g. Fermi Teslas don't have temp. sensors), and one process for reporting the progress. The GPU processes each allocate 90% of the free GPU memory, initialize 2 random 2048*2048 matrices, and continuously perform efficient CUBLAS matrix-matrix multiplication routines on them and store the results across the allocated memory. Both floats and doubles are supported. Correctness of the calculations is checked by comparing results of new calculations against a previous one -- on the GPU. This way the GPUs are 100% busy all the time and CPUs idle. The number of erroneous calculations is brought back to the CPU and reported to the user along with the number of operations performed so far and the GPU temperatures.

Real-time progress and summaries every ~10% are printed as shown below. Matrices processed are cumulative, whereas errors are for that summary. GPUs are separated by slashes. The program exits with a conclusion after it has been running for the number of seconds given as the last command line parameter. If you want to burn using doubles instead, give parameter "-d" before the burn duration. The example below was on a machine that had one working GPU and one faulty (too much factory overclocking and thus slightly unstable (you wouldn't have noticed it during gaming)):
# gpu-burn
Multi-GPU CUDA stress test
http://wili.cc/blog/gpu-burn.html
 burn with floats for an hour: make && ./gpu_burn 3600
If you're running a Tesla, burning with doubles instead stresses the card more (as it was friendly pointed out to me in the comments by Rick W): make && ./gpu_burn -d 3600
You might have to show the Makefile to your CUDA if it's not in the default path, and also to a version of gcc your nvcc can work with. It expects to find nvidia-smi from your default path. 
