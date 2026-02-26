# Graphcast on Nvidia Arm + GPU

## 25th Feb 2026 (late evening)

I've been trying to get a container built on Nvidia ARM + GPU, specifically both PGX GB10s on loan from Lenovo and Locust, our GH200.

There's a fundamental stumbling block, that ecmwflibs (https://github.com/ecmwf/ecmwflibs/) does not provide wheels for any verson of Python on Arm and is quite complicated to build wheels for with a lot of external dependencies and ecmwflibs is a dependency for climetlab.

See https://github.com/ecmwf/ecmwflibs/blob/master/scripts/build-linux.sh for examples of how many steps are needed to do it in their CI/CD pipeline.

(I am not alone, there are dozens of us https://github.com/ecmwf/ecmwflibs/issues/34)

Now technically we don't need climetlab or ecmwflibs to run Graphcast - we only need them to run it though the ECMWF framework and so I think on these platforms we are forced to run without it.

## 26th Feb 2026

I've adapted the Nvidia Docker to not include the ECMWF stuff and decided to port the "basic example" notebook from the Graphcast Repo to "not a notebook" just to prove things work. This is `Docker/Nvidia/examples/example.py`.

This container has some additional packages to support this example, specifically ipywidgets and support for Google cloud storage.

This example seems to work correctly in the container.
