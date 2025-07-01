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
[uccaoke@ip-10-134-25-2 ~]$ podman run -it --rm --group-add keep-groups --device /dev/kfd:rwm --device /dev/dri:rwm --ipc=host -v $HOME/podmanhome:/home/uccaoke:Z docker.io/rocm/jax:rocm6.3.2-jax0.4.31-py3.12 /bin/bash -l
root@0e529ba2cd6d:/# python3
Python 3.12.3 (main, Jan 17 2025, 18:03:48) [GCC 13.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import jax
>>> jax.devices()
[RocmDevice(id=0), RocmDevice(id=1), RocmDevice(id=2), RocmDevice(id=3), RocmDevice(id=4), RocmDevice(id=5), RocmDevice(id=6), RocmDevice(id=7)]
>>> from jaxlib import gpu_triton
>>> gpu_triton._hip_triton
<module 'jax_rocm60_plugin._triton' from '/opt/venv/lib/python3.12/site-packages/jax_rocm60_plugin/_triton.so'>
>>>
root@0e529ba2cd6d:/# cd home/uccaoke/
root@0e529ba2cd6d:/home/uccaoke# ls
jaxpi.py  protein_shake  ucnvtwe_dplm
root@0e529ba2cd6d:/home/uccaoke# python3 jaxpi.py
Estimating Pi with:
  1600000 slices
  1 devices(s)

Estimated value of Pi: 3.141592502593994
Time taken: 7.941874980926514 seconds.

root@0e529ba2cd6d:/home/uccaoke# python3 jaxpi.py 16000000
Estimating Pi with:
  16000000 slices
  1 devices(s)

Estimated value of Pi: 3.141592502593994
Time taken: 7.734260320663452 seconds.

root@0e529ba2cd6d:/home/uccaoke#
logout
```

Which is promising.

The plan: build a Docker container with everything we need.