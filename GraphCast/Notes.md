# Graphcast on ATLAS

A while back, I got Graphcast working on Myriad for some researchers.

This is my attempt to get it running on ATLAS.

Things I remember from my notes on Myriad.

1. Numpy < 2
2. JAX <= 0.4.23 due to...
3. chex==0.1.83
4. Repo at https://github.com/deepmind/graphcast.git
5. Need lots of extra libraries and config files to get datasets.
6. Bug in earthkit-data fixed by installing 0.11.1

Things for ATLAS:

1. We have containers for various versions of JAX on ROCm.
2. JAX 0.4.23 is quite old but:

```
[uccaoke@ip-10-134-25-2 ~]$ podman run -it --rm --group-add keep-groups --device /dev/kfd:rwm --device /dev/dri:rwm --ipc=host -v $HOME/podmanhome:/home/uccaoke:Z docker.io/rocm/jax:rocm6.0.0-jax0.4.23-py3.11.0 /bin/bash -l
root@3c9d74dc29d9:/# python3
Python 3.11.0 (main, Jan  6 2024, 15:35:04) [GCC 9.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import jax
2025-07-01 15:55:23.209994: E external/xla/xla/stream_executor/plugin_registry.cc:90] Invalid plugin kind specified: DNN
>>> jax.devices()
[rocm(id=0), rocm(id=1), rocm(id=2), rocm(id=3), rocm(id=4), rocm(id=5), rocm(id=6), rocm(id=7)]
>>> from jaxlib import gpu_triton
>>> gpu_triton._hip_triton
<module 'jaxlib.rocm._triton' from '/pyenv/versions/3.11.0/lib/python3.11/site-packages/jaxlib/rocm/_triton.so'>
>>>
root@3c9d74dc29d9:/# cd /home/uccaoke
root@3c9d74dc29d9:/home/uccaoke# python3 jaxpi.py
2025-07-01 15:56:06.655861: E external/xla/xla/stream_executor/plugin_registry.cc:90] Invalid plugin kind specified: DNN
Estimating Pi with:
  1600000 slices
  1 devices(s)

2025-07-01 15:56:10.121443: W external/xla/xla/service/gpu/gpu_compiler.cc:549] GpuCompilationEnvironment of hlo_module jit__linspace:
2025-07-01 15:56:10.121497: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_backend_optimization_level: 3
2025-07-01 15:56:10.121508: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_eliminate_hlo_implicit_broadcast: true
2025-07-01 15:56:10.121517: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_cpu_multi_thread_eigen: true
2025-07-01 15:56:10.121524: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_cuda_data_dir: "./cuda_sdk_lib"
2025-07-01 15:56:10.121533: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_llvm_enable_alias_scope_metadata: true
2025-07-01 15:56:10.121543: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_llvm_enable_noalias_metadata: true
2025-07-01 15:56:10.121551: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_llvm_enable_invariant_load_metadata: true
2025-07-01 15:56:10.121559: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_force_host_platform_device_count: 1
2025-07-01 15:56:10.121566: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_cpu_fast_math_honor_nans: true
2025-07-01 15:56:10.121574: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_cpu_fast_math_honor_infs: true
2025-07-01 15:56:10.121581: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_allow_excess_precision: true
2025-07-01 15:56:10.121589: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_autotune_level: 4
2025-07-01 15:56:10.121597: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_cpu_fast_math_honor_division: true
2025-07-01 15:56:10.121606: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_cpu_fast_math_honor_functions: true
2025-07-01 15:56:10.121614: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_dump_max_hlo_modules: -1
2025-07-01 15:56:10.121622: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_multiheap_size_constraint_per_heap: -1
2025-07-01 15:56:10.121630: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_enable_async_all_reduce: true
2025-07-01 15:56:10.121639: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_strict_conv_algorithm_picker: true
2025-07-01 15:56:10.121647: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_all_reduce_combine_threshold_bytes: 31457280
2025-07-01 15:56:10.121656: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_enable_cudnn_frontend: true
2025-07-01 15:56:10.121664: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_nccl_termination_timeout_seconds: -1
2025-07-01 15:56:10.121673: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_enable_shared_constants: true
2025-07-01 15:56:10.121681: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_redzone_scratch_max_megabytes: 4096
2025-07-01 15:56:10.121689: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_simplify_all_fp_conversions: true

<snip>

2025-07-01 15:56:11.137515: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_cpu_fast_math_honor_functions: true
2025-07-01 15:56:11.137524: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_dump_max_hlo_modules: -1
2025-07-01 15:56:11.137532: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_multiheap_size_constraint_per_heap: -1
2025-07-01 15:56:11.137541: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_enable_async_all_reduce: true
2025-07-01 15:56:11.137550: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_strict_conv_algorithm_picker: true
2025-07-01 15:56:11.137559: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_all_reduce_combine_threshold_bytes: 31457280
2025-07-01 15:56:11.137568: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_enable_cudnn_frontend: true
2025-07-01 15:56:11.137576: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_nccl_termination_timeout_seconds: -1
2025-07-01 15:56:11.137584: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_enable_shared_constants: true
2025-07-01 15:56:11.137592: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_redzone_scratch_max_megabytes: 4096
2025-07-01 15:56:11.137601: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_simplify_all_fp_conversions: true
2025-07-01 15:56:11.137610: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_enable_xla_runtime_executable: true
2025-07-01 15:56:11.137619: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_shape_checks: RUNTIME
2025-07-01 15:56:11.137627: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_normalize_layouts: true
2025-07-01 15:56:11.137635: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_cpu_enable_mlir_tiling_and_fusion: true
2025-07-01 15:56:11.137644: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_dump_enable_mlir_pretty_form: true
2025-07-01 15:56:11.137653: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_enable_triton_gemm: true
2025-07-01 15:56:11.137661: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_enable_cudnn_int8x32_convolution_reordering: true
2025-07-01 15:56:11.137671: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_cpu_enable_experimental_deallocation: true
2025-07-01 15:56:11.137681: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_cpu_enable_mlir_fusion_outlining: true
2025-07-01 15:56:11.137690: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_cpu_matmul_tiling_m_dim: 8
2025-07-01 15:56:11.137698: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_cpu_matmul_tiling_n_dim: 8
2025-07-01 15:56:11.137708: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_cpu_matmul_tiling_k_dim: 8
2025-07-01 15:56:11.137717: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_graph_num_runs_to_instantiate: -1
2025-07-01 15:56:11.137726: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_lhs_enable_gpu_async_tracker: true
2025-07-01 15:56:11.137735: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_collective_inflation_factor: 1
2025-07-01 15:56:11.137744: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_graph_min_graph_size: 5
2025-07-01 15:56:11.137754: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_enable_reassociation_for_converted_ar: true
2025-07-01 15:56:11.137763: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_all_gather_combine_threshold_bytes: 31457280
2025-07-01 15:56:11.137774: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_reduce_scatter_combine_threshold_bytes: 31457280
2025-07-01 15:56:11.137783: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_enable_highest_priority_async_stream: true
2025-07-01 15:56:11.137791: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_auto_spmd_partitioning_memory_budget_ratio: 1.1
2025-07-01 15:56:11.137799: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_redzone_padding_bytes: 8388608
2025-07-01 15:56:11.137808: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_triton_fusion_level: 2
2025-07-01 15:56:11.137816: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_graph_eviction_timeout_seconds: 60
2025-07-01 15:56:11.137823: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_enable_gpu2_hal: true
2025-07-01 15:56:11.137831: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_copy_insertion_use_region_analysis: true
2025-07-01 15:56:11.137838: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_collective_permute_decomposer_threshold: 9223372036854775807
2025-07-01 15:56:11.137846: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_enable_split_k_autotuning: true
2025-07-01 15:56:11.137854: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_enable_reduction_epilogue_fusion: true
2025-07-01 15:56:11.137861: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_cublas_fallback: true
2025-07-01 15:56:11.137869: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_filter_kernels_spilling_registers_on_autotuning: true
2025-07-01 15:56:11.137877: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_debug_buffer_assignment_show_max: 15
2025-07-01 15:56:11.137884: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_enable_dumping: true
2025-07-01 15:56:11.137892: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_enable_all_gather_combine_by_dim: true
2025-07-01 15:56:11.137899: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_enable_reduce_scatter_combine_by_dim: true
2025-07-01 15:56:11.137907: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_enable_command_buffer: FUSION
2025-07-01 15:56:11.137915: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_enable_command_buffer: CUBLAS
2025-07-01 15:56:11.137922: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_enable_cub_radix_sort: true
2025-07-01 15:56:11.137930: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_memory_limit_slop_factor: 95
2025-07-01 15:56:11.137937: W external/xla/xla/service/gpu/gpu_compiler.cc:549] xla_gpu_threshold_for_windowed_einsum_mib: 100000
Estimated value of Pi: 3.141592502593994
Time taken: 4.384366750717163 seconds.

```

Which is sort of promising.

The plan: build a Docker container with everything we need.

Issues are: when I looked this needed this very old version of JAX and I see a *lot* of bugs when I search for some of these errors.

OK - built container;

```
AttributeError: module 'jax.stages' has no attribute 'OutInfo'
```

Ooooo.

Also it looks like Graphcast has been updatd.

So actually we might not need the ancient JAX...