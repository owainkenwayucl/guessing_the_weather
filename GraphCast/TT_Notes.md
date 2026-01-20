# Here are the voyages of trying to get Graphcast running on Tenstorrent.

It looks like the master/main branch of Graphcast has been updated to work on JAX >= 0.7.0 which is good because the current TT JAX I have is 0.7.1.

So let's take the `tt-xla-slim` image and install the packages into it that we installed for the Myriad install.
