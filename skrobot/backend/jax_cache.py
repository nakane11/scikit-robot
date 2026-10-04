"""Persistent JAX compilation cache.

JIT-compiling a batch IK solver or a trajectory problem takes from a
fraction of a second to tens of seconds, and it is paid again in every new
process. JAX can keep the compiled executables on disk and reuse them; this
module switches that on with settings suited to scikit-robot.

A cache entry is looked up by the traced computation, so it is only hit when
the same computation is traced again. ``extract_fk_parameters`` rounds the
constants it bakes into the graph for that reason: without it the entry
written by one process can miss in the next one although nothing changed.
"""

import os


ENV_CACHE_DIR = 'SKROBOT_JAX_CACHE_DIR'


def default_cache_dir():
    """Return the directory used when none is given.

    The first of these that is set wins: ``$SKROBOT_JAX_CACHE_DIR``,
    ``$JAX_COMPILATION_CACHE_DIR`` (JAX's own variable), then
    ``$XDG_CACHE_HOME/skrobot/jax_compilation_cache`` (``~/.cache`` when
    ``XDG_CACHE_HOME`` is unset).
    """
    for name in (ENV_CACHE_DIR, 'JAX_COMPILATION_CACHE_DIR'):
        path = os.environ.get(name)
        if path:
            return os.path.expanduser(path)
    base = os.environ.get('XDG_CACHE_HOME') or os.path.join('~', '.cache')
    return os.path.expanduser(
        os.path.join(base, 'skrobot', 'jax_compilation_cache'))


def enable_persistent_cache(cache_dir=None,
                            min_compile_time_secs=0.0,
                            min_entry_size_bytes=0):
    """Keep JAX's compiled executables on disk across processes.

    Call it once, before the first JIT compilation you want cached. It
    works whether or not JAX has already been imported. Anything compiled
    before the call is not written to the cache.

    Parameters
    ----------
    cache_dir : str, optional
        Directory of the cache. It is created when missing. See
        :func:`default_cache_dir` for the default.
    min_compile_time_secs : float
        Only computations that took at least this long to compile are
        cached. JAX's own default is 1 second, which leaves the many small
        solvers out; the default here is 0 (cache everything).
    min_entry_size_bytes : int
        Only executables at least this large are cached. JAX's own default
        is 0 as well.

    Returns
    -------
    str
        The cache directory in use.

    Raises
    ------
    ImportError
        If JAX is not installed.

    Examples
    --------
    >>> from skrobot.backend import enable_persistent_cache
    >>> enable_persistent_cache()  # doctest: +SKIP
    """
    import jax

    cache_dir = os.path.abspath(
        os.path.expanduser(cache_dir) if cache_dir else default_cache_dir())
    os.makedirs(cache_dir, exist_ok=True)

    jax.config.update('jax_compilation_cache_dir', cache_dir)
    jax.config.update('jax_persistent_cache_min_compile_time_secs',
                      float(min_compile_time_secs))
    try:
        jax.config.update('jax_persistent_cache_min_entry_size_bytes',
                          int(min_entry_size_bytes))
    except AttributeError:
        # Older JAX releases have no such option; every entry is cached.
        pass
    return cache_dir
