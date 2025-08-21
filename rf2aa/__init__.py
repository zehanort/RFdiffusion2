import contextlib
import os

projdir = os.path.realpath(os.path.dirname(__file__))
from ipd.dev import lazyimport, install_ipd_pre_commit_hook
import rf2aa.sym.rf2_sym_manager  # noqa needed for registration

model = lazyimport('rf2aa.model')
motif = lazyimport('rf2aa.motif')
sym = lazyimport('rf2aa.sym')
tests = lazyimport('rf2aa.tests')
tools = lazyimport('rf2aa.tools')
util = lazyimport('rf2aa.util')

with contextlib.suppress(ImportError):
    from icecream import ic
    ic.configureOutput(includeContext=True)

install_ipd_pre_commit_hook(projdir, '..')
