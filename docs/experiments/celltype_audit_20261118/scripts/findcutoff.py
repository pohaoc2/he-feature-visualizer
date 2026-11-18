# Python port of the Lin et al. 2023 (Nature Cancer, 10.1038/s43018-023-00576-1) marker gating, labsyspharm/orion-crc:
#   CyCIF_scripts/CycIF_findcutoff.m   K-component GMM on log intensity, cutoff where the negative-component share of cells
#                                      above it (Matlab_scripts/falserate.m) is closest to the target FDR
#   CyCIF_scripts/CycIF_visualgate_Shu.m  log, 50 000-cell sample without replacement, findcutoff(minval=2), gate applied to all
#   CyCIF_scripts/CycIF_setgate.m      positive = intensity > exp(gate)
# fitgmdist(...,'replicates',10) -> sklearn GaussianMixture(n_init=10); MATLAB prctile -> numpy method "hazen".
import numpy as np
from scipy.stats import norm
from sklearn.mixture import GaussianMixture


def prctile(x, p):
    """MATLAB prctile (midpoint rule, percentile i of n sorted values at 100 (i - 0.5) / n)."""
    return float(np.percentile(np.asarray(x, float), p, method="hazen"))


def colon(a, step, b):
    """MATLAB a:step:b."""
    n = int(np.floor((b - a) / step + 1e-10)) + 1
    return a + step * np.arange(max(n, 0))


def falserate(x, mu, sd, w):
    """Matlab_scripts/falserate.m: share of cells above x drawn from the lowest-mean component (vs the highest)."""
    lo, hi = int(np.argmin(mu)), int(np.argmax(mu))
    a = w[lo] * (1 - norm.cdf(x, mu[lo], sd[lo])); b = w[hi] * (1 - norm.cdf(x, mu[hi], sd[hi]))
    return a / (b + a)


def fit_gmm(x, k, seed):
    g = GaussianMixture(k, covariance_type="full", n_init=10, tol=1e-6, max_iter=1000, random_state=seed).fit(x.reshape(-1, 1))
    return g.means_.ravel(), np.sqrt(g.covariances_.ravel()), g.weights_


def findcutoff(markersample, minval=2.0, k=2, fdr=0.05, seed=0):
    """CycIF_findcutoff.m. markersample = log intensities. Returns (cutoff, poscells, (mu, sd, w))."""
    x = np.asarray(markersample, float)
    mu, sd, w = fit_gmm(x[x > minval], k, seed)
    order = np.argsort(mu); minid, maxid = order[0], order[1]          # as in the .m: 2nd-lowest mean when K > 2
    peak_min, peak_plus = norm.pdf(0, 0, sd[minid]), norm.pdf(0, 0, sd[maxid])
    if w[maxid] * peak_plus > w[minid] * peak_min:
        search = colon(prctile(x, 2), 0.01, prctile(x, 70))
    else:
        search = colon(prctile(x, 10), 0.01, prctile(x, 98))
    obj = (falserate(search, mu, sd, w) - fdr) ** 2
    cutoff = float(search[int(np.argmin(obj))])
    return cutoff, x > cutoff, (mu, sd, w)


def visualgate_shu(values, fdr, k=2, n_sample=50000, seed=0):
    """CycIF_visualgate_Shu.m: gate fitted on a 50 000-cell sample, applied to all cells. Returns (gate, positive)."""
    v = np.asarray(values, float); rng = np.random.default_rng(seed)
    s = v[rng.choice(len(v), n_sample, replace=False)] if len(v) > n_sample else v
    with np.errstate(divide="ignore"):
        gate, _, _ = findcutoff(np.log(s), 2.0, k, fdr, seed)
    return gate, v > np.exp(gate)
