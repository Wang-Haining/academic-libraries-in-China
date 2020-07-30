import pymc3 as pm
import numpy as np
import pandas as pd

expenditureTotalAdjusted_melted = pd.read_pickle('./ETAM.pickle')

with pm.Model() as m6:
    # Hyperpriors for group nodes
    mu_a = pm.Normal('mu_a', mu=0., sigma=10)
    sigma_a = pm.HalfNormal('sigma_a', 5.)
    mu_b = pm.Normal('mu_b', mu=0., sigma=10)
    sigma_b = pm.HalfNormal('sigma_b', 5.)
    mu_c = pm.Normal('mu_c', mu=0., sigma=10)
    sigma_c = pm.HalfNormal('sigma_c', 5.)
    mu_d = pm.Normal('mu_d', mu=0., sigma=10)
    sigma_d = pm.HalfNormal('sigma_d', 5.)
    mu_e = pm.Normal('mu_e', mu=0., sigma=10)
    sigma_e = pm.HalfNormal('sigma_e', 5.)
    mu_f = pm.Normal('mu_f', mu=0., sigma=10)
    sigma_f = pm.HalfNormal('sigma_f', 5.)
    mu_g = pm.Normal('mu_g', mu=0., sigma=10)
    sigma_g = pm.HalfNormal('sigma_g', 5.)

    a = pm.Normal('a', mu=mu_a, sigma=sigma_a, shape=1600)
    b = pm.Normal('b', mu=mu_b, sigma=sigma_b, shape=1600)
    c = pm.Normal('c', mu=mu_c, sigma=sigma_c, shape=1600)
    d = pm.Normal('d', mu=mu_d, sigma=sigma_d, shape=1600)
    e = pm.Normal('e', mu=mu_e, sigma=sigma_e, shape=1600)
    f = pm.Normal('f', mu=mu_f, sigma=sigma_f, shape=1600)
    g = pm.Normal('g', mu=mu_g, sigma=sigma_g, shape=1600)
    # Model error
    sigma = pm.HalfCauchy('sigma', 5.)

    y = a[expenditureTotalAdjusted_melted.code.values] + \
        b[expenditureTotalAdjusted_melted.code.values] * expenditureTotalAdjusted_melted.zyear.values + \
        c[expenditureTotalAdjusted_melted.code.values] * np.power(expenditureTotalAdjusted_melted.zyear.values, 2) + \
        d[expenditureTotalAdjusted_melted.code.values] * np.power(expenditureTotalAdjusted_melted.zyear.values, 3) + \
        e[expenditureTotalAdjusted_melted.code.values] * np.power(expenditureTotalAdjusted_melted.zyear.values, 4) + \
        f[expenditureTotalAdjusted_melted.code.values] * np.power(expenditureTotalAdjusted_melted.zyear.values, 5) + \
        g[expenditureTotalAdjusted_melted.code.values] * np.power(expenditureTotalAdjusted_melted.zyear.values, 6)
    # Data likelihood
    likelihood = pm.Normal('likelihood', mu=y, sigma=sigma, observed=expenditureTotalAdjusted_melted.zvalue.values)

with m6:
    m6_trace = pm.sample(20000, tune=20000)

pm.save_trace(m6_trace, directory="./m6_trace")