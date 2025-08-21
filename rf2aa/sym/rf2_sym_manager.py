import torch as th
import rf2aa as rf
import ipd
import logging

log = logging.getLogger(__name__)

class RF2SymmetryManager(ipd.sym.SymmetryManager):
    """
    Implements default rf symmetry operations.

    This class is the default symmetry manager for rf. It implements the apply_symmetry method, which is the main entry point for applying symmetry to any object. The object can be a sequence, coordinates, or a pair xyz/pair. The object will be passed to the appropriate method based on its type and shape. The method will be called with the object and all relevant symmetry parameters. The method should return the object with symmetry applied. If the object is a pair xyz,pair, the method should return a tuple of xyz,pair. If the object is a 'sequence', the method should return the sequence with the asu copies to the symmetric subs. 'sequence' can be anything with shape that starts with L
    """
    kind = 'rf2aa'

    def init(self, *a, idx=None, **kw):
        '''Create an RF2SymmetryManager'''
        super().init(*a, **kw)
        if self.symid.lower() == 'input_defined': return
        self.symmatrix, self._symmRs, self.metasymm, _ = rf.sym.symm_subunit_matrix(self.symid, self.opt)
        self.symmatrix = self.symmatrix.to(self.device)
        self._symmRs = self._symmRs.to(self.device)
        self.metasymm = [[x.to(self.device) for x in self.metasymm[0]], self.metasymm[1]]
        # self.asucenvec = self.asucenvec.to(self.device)
        # ic(self.symid, self.opt.max_nsub)
        self.asucen = th.as_tensor(ipd.sym.canonical_asu_center(self.symid)[:3], device=self.device)
        self.asucenvec = ipd.h.normalized(self.asucen)
        if 'nsub' in self.opt and self.opt.nsub:
            # assert int(self.metasymm[1][0]) == self.opt.nsub
            if self.opt.has('Lasu'):
                self.opt.L = self.opt.Lasu * self.opt.nsub
            elif self.opt.has('repeat_length'):
                self.opt.L = self.opt.repeat_length * self.opt.nsub
                assert self.opt.L % self.opt.repeat_length == 0
                self.opt.Lasu = self.opt.L // self.opt.nsub
        self.opt.nsub = int(self.metasymm[1][0])
        self.symmsub = th.arange(self.nsub).to(self.device)
        if self.symid.startswith('I') and self.symid != 'I':
            self._full_symmetry = ipd.sym.frames('I', torch=True)[:, :3, :3].contiguous()
            self._full_symmetry = self._full_symmetry.to(self.device).to(th.float32)

    def apply_symmetry(self, xyz, pair, Lasu, opts, update_symmsub=False, disable_all_fitting=False, **_):
        '''Apply symmetry to an object or xyz/pair'''
        opts.disable_all_fitting = disable_all_fitting
        xyz = ipd.sym.asu_to_best_frame_if_necessary(self, xyz, **opts)
        xyz = ipd.sym.set_particle_radius_if_necessary(self, xyz, **opts)
        xyz = ipd.sym.asu_to_canon_if_necessary(self, xyz, **opts)
        s = self.idx

        if update_symmsub and not disable_all_fitting:
            xyz, pair, self.symmsub = ipd.sym.update_symm_subs_track_module(xyz[None], pair[None], self.symmatrix,
                                                                            self.symmsub, self.allsymmRs, self.metasymm,
                                                                            opts)
            return xyz, pair
        else:
            oshape = xyz.shape
            xyz = xyz.reshape(oshape[0], -1, 3)
            xyz = ipd.sym.update_symm_Rs(xyz[None], Lasu, self.symmsub, self.allsymmRs, opts)
            return xyz.reshape(oshape)

ipd.sym.set_default_sym_manager('rf2aa')
