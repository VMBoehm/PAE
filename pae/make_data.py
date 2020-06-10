from scipy.stats import special_ortho_group
import numpy as np

class Banana():

    def __init__(self,Q=0.01,D=32,lower=-15,upper=15,random_rot=True,randseed=0):
        
        self.Q     = Q
        self.D     = D
        self.lower = lower
        self.upper = upper
        self.randseed = randseed
        np.random.seed(randseed)
        
        self.A = special_ortho_group.rvs(D)
        self.random_rot= random_rot

        ir = np.array((np.full(self.D, self.lower),
               np.full(self.D, self.upper))).T
        self.bound = ir

    def logp(self,xx):
        xx = xx @ self.A.T
        return -np.sum((xx[..., ::2]**2 - xx[..., 1::2])**2 / self.Q +
                   (xx[..., ::2] - 1)**2, axis=-1) + 18.526    #normalization constant from He et al

    def grad(self,xx):
        xx = xx @ self.A.T
        _pfpx2i1 = 2 * (xx[..., 1::2] - xx[..., ::2]**2) / Q
        _pfpx2i = 2 * (xx[..., ::2] - 1) - 2 * xx[..., ::2] * _pfpx2i1
        res = np.empty_like(xx)
        res[..., ::2] = _pfpx2i
        res[..., 1::2] = _pfpx2i1
        return -res @ self.A

    def _in_bound(self,xx):
        xxt = np.atleast_2d(xx).T
        return np.product([np.where(xi>self.bound[i,0], True, False) *
                       np.where(xi<self.bound[i,1], True, False) for i, xi in
                       enumerate(xxt)], axis=0).astype(bool)

    def generate_samples(self,N):
        np.random.seed(self.randseed)
        data = np.random.randn(N, self.D) * 0.5**0.5
        data[:, ::2] += 1
        data[:, 1::2] = data[:, ::2]**2 + data[:, 1::2] * self.Q**0.5
        if self.random_rot:
            data = data @ self.A
        data = data[self._in_bound(data)]
        return data.T

