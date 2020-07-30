import pymc3 as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read in data
expenditureTotalAdjusted_melted = pd.read_pickle('./remote/ETAM.pickle')

# transfer data to float to make it more computationally efficient
expenditureTotalAdjusted_melted.school = expenditureTotalAdjusted_melted.school.astype('string')
expenditureTotalAdjusted_melted.code = expenditureTotalAdjusted_melted.code.astype('category')
expenditureTotalAdjusted_melted.zyear = expenditureTotalAdjusted_melted.zyear.astype('float64')
expenditureTotalAdjusted_melted.zvalue = expenditureTotalAdjusted_melted.zvalue.astype('float64')
n_libraries = len(expenditureTotalAdjusted_melted.school.unique())
index = np.array(expenditureTotalAdjusted_melted.code)
# data is ready to be fed into the model

with pm.Model() as m1:
    #     index = np.array(expenditureTotalAdjusted_melted.code)
    # Hyperpriors for group nodes
    mu_a = pm.Normal('mu_a', mu=0., sigma=10)
    sigma_a = pm.HalfNormal('sigma_a', 5.)
    mu_b = pm.Normal('mu_b', mu=0., sigma=10)
    sigma_b = pm.HalfNormal('sigma_b', 5.)

    # Intercept for each county, distributed around group mean mu_a
    # Above we just set mu and sd to a fixed value while here we
    # plug in a common group distribution for all a and b (which are
    # vectors of length n_counties).
    a = pm.Normal('a', mu=mu_a, sigma=sigma_a, shape=n_libraries)
    # Intercept for each county, distributed around group mean mu_a
    b = pm.Normal('b', mu=mu_b, sigma=sigma_b, shape=n_libraries)

    # underlining trend for individual library
    sigma = pm.HalfCauchy('sigma', 5.)
    nu = pm.HalfCauchy('nu', 5.)

    mu = a[index] + b[index] * expenditureTotalAdjusted_melted.zyear

    # Data likelihood
    likelihood = pm.StudentT('likelihood', mu=mu,
                             sigma=sigma, nu=nu,
                             observed=expenditureTotalAdjusted_melted.zvalue)

    with m1:
        m1_trace = pm.sample(50000, tune=50000, cores=4, init='auto')

    # pm.save_trace(m1_trace, directory="./m1_trace")
    with m1:
        m1_trace = pm.load_trace(directory="./remote/m1_trace")
    # pm.plot_posterior(m1_trace)
    # plt.show()

"""

"""

with pm.Model() as m2:
    #     index = np.array(expenditureTotalAdjusted_melted.code)
    # Hyperpriors for group nodes
    mu_a = pm.Normal('mu_a', mu=0, sigma=10)
    sigma_a = pm.HalfNormal('sigma_a', 10)
    mu_b = pm.Normal('mu_b', mu=0, sigma=10)
    sigma_b = pm.HalfNormal('sigma_b', 10)
    mu_c = pm.Normal('mu_c', mu=0, sigma=10)
    sigma_c = pm.HalfNormal('sigma_c', 10)

    # Intercept for each county, distributed around group mean mu_a
    # Above we just set mu and sd to a fixed value while here we
    # plug in a common group distribution for all a and b (which are
    # vectors of length n_counties).
    a = pm.Normal('a', mu=mu_a, sigma=sigma_a, shape=n_libraries)
    # Intercept for each county, distributed around group mean mu_a
    b = pm.Normal('b', mu=mu_b, sigma=sigma_b, shape=n_libraries)
    c = pm.Normal('c', mu=mu_c, sigma=sigma_c, shape=n_libraries)

    # underlining trend for individual library
    sigma = pm.HalfNormal('sigma', 10)
    nu = pm.HalfNormal('nu', 10)

    mu = a[index] + b[index] * expenditureTotalAdjusted_melted.zyear + \
                    c[index] * np.power(expenditureTotalAdjusted_melted.zyear, 2)

    # Data likelihood
    likelihood = pm.StudentT('likelihood', mu=mu,
                             sigma=sigma, nu=nu,
                             observed=expenditureTotalAdjusted_melted.zvalue)

    with m2:
        m2_trace = pm.sample(10000, tune=10000, cores=4, init='auto')

    pm.save_trace(m2_trace, directory="./m2_trace")
