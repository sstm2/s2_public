import numpy as np
import pandas as pd

# use to initialize r and q
def init_mse_pars(y, x):
    q = np.nanmean((x - y) ** 2)
    if np.isnan(q):
        q = 1
    return q


def init_pars(y, x):
    idx = [np.isfinite(y[j]) and np.isfinite(x[j]) for j in range(len(y))]  #  clunk
    if not any(idx):
        b = 1
        q = 1
    else:
        b = np.sum(y[idx] * x[idx]) / np.sum(x[idx] ** 2)
        q = np.mean((y[idx] - b * x[idx]) ** 2)
    return b, q


def fit_series(Y, lhs=None, tolerance=1e-6, weights=None, beta_in=None):
    """Fit RHS series to a target variable

    Estimate a series. Designed for extending a timeseries using multiple overlapping series, but useful for lots of
    things. If lhs is None, the model will fit the first series in Y.
    If lhs is specified, the model will use Y to estimate the lhs variable. The idea is
    similar to OLS, but allows for missing data and uses dimension reduction to avoid overfitting. Models are estimated
    via the EM algorithm.

    Args:
        Y (array-like): If lhs is None, the target variable is the first series in Y.
           All columns must be numeric.
        lhs (array-like, optional): If lhs is specified, it will be the target variable. Must be numeric.
        tolerance (float): difference in likelihood needed to break iterations (smaller tolerance -> more precision)
        weights (array-like, optional): A series or array specifying the weight on each observation. These need not
            sum to one.
        beta_in (array-like): Fixed OLS pars (ie do not estimate)

    Returns:
        {
        array: x is an array of fitted values equal to the target series when the target series is observed,
        array: x_fit is an array of fitted values that does not use the target series to allow evaluation of in-sample,
                fit
        array: beta is the weights used for each RHS variable when all RHS variables are observed,
        }
    """

    if isinstance(Y, pd.DataFrame):
        Y = Y.to_numpy()
    if lhs is not None:
        if isinstance(lhs, pd.DataFrame) or isinstance(lhs, pd.Series):
            lhs = lhs.to_numpy()
        Y = np.column_stack((lhs, Y))
    k = np.shape(Y)[1]  # number of series
    # scale = np.ones(k)
    # center = np.zeros(k)
    # if scale_input:
    #     for j in range(k):  # make Y reasonable numbers
    #         center[j] = np.nanmean(Y[:, j])
    #         Y[:, j] -= center[j]
    #         scale[j] = np.nanstd(Y[:, j])
    #         Y[:, j] /= scale[j]
    if weights is not None:
        if len(weights) != np.shape(Y)[0]:
            raise ValueError("Length of weights and shape of input data do not agree")
        else:
            weights = weights / weights.mean()
            for j in range(k):
                Y[:, j] = weights * Y[:, j]

    b = np.ones(k)  # initial OLS parameters
    h = np.ones(k)  # initial Kfilter parameters
    T = np.shape(Y)[0]
    x = np.zeros(T)  # vector to fill in with estimated x values
    p = np.zeros(T)  # vector to fill in with estimated p values
    r = np.zeros(k)  # initial guess for variance
    q = np.zeros(k)  # initial guess for variance
    # weights = np.arange(T)**2
    if beta_in is not None:
        if isinstance(beta_in, pd.Series):
            beta_in = beta_in.to_numpy()
        if len(beta_in) == k - 1:
            beta_in = np.concatenate([np.array([1]), beta_in])  # add a 1 for the lhs series
        elif len(beta_in) != k:
            raise ValueError("Shape of 'beta_in' does not agree with shape of 'Y'")
        b = beta_in  # OLS pars
        h = 1 / beta_in  # Kfilter pars
        for j in range(1, k):
            r[j] = init_mse_pars(Y[:, j], h[j] * Y[:, 0])
            q[j] = init_mse_pars(Y[:, 0], b[j] * Y[:, j])
    else:
        for j in range(1, k):
            h[j], r[j] = init_pars(Y[:, j], h[j] * Y[:, 0])
            b[j], q[j] = init_pars(Y[:, 0], b[j] * Y[:, j])
    it = 0
    tol = 1
    ll_old = -1e20
    # (2) Filter
    while tolerance < tol and it < 1000:
        ll = 0
        for t in range(T):
            y_obs = Y[t, :]
            is_finite = np.isfinite(y_obs)
            y_obs = y_obs[is_finite]
            b_obs = b[is_finite]  # OLS pars
            h_obs = h[is_finite]  # Kfilter pars
            r_obs = r[is_finite]  # Kfilter error
            q_obs = q[is_finite]  # OLS error
            if is_finite[0]:  # if target is observed
                x[t] = y_obs[0]
            elif np.all(~is_finite):  # if nothing is observed
                x[t] = np.nan
            else:
                xt = b_obs[0] * y_obs[0]  # prior (based on observation of series 0)
                pt = q_obs[0]  # prior variance
                for j in range(1, len(y_obs)):
                    s = h_obs[j] * pt * h_obs[j] + r_obs[j]  # update variance
                    c = h_obs[j] * pt  # update covariance
                    nu = y_obs[j] - h_obs[j] * xt  # prediction error
                    xt += (c / s) * nu  # update based on observation of series j
                    pt -= (c**2) / s  # new variance of updated xt
                    ll -= (np.log(2 * np.pi) + np.log(s) + (nu * nu / s)) / 2  # loglike
                x[t] = xt
                p[t] = pt
        # Estiamte pars
        for j in range(1, k):
            is_finite = np.isfinite(Y[:, j])
            y_obs = Y[is_finite, j]
            if beta_in is None:  # only estimate if parameters not specified
                b[j] = np.sum(x[is_finite] * y_obs) / np.sum(np.square(y_obs))  # OLS pars
                h[j] = np.sum(x[is_finite] * y_obs) / (np.sum(np.square(x[is_finite])) + np.sum(p))  # Kfilter pars
            r[j] = np.mean(np.square(y_obs - h[j] * x[is_finite])) + np.mean(h[j] * p[is_finite] * h[j])
            q[j] = np.mean(np.square(x[is_finite] - b[j] * y_obs)) + np.mean(p[is_finite])
        # if ll == 0:  # all lhs variables observed, or no multiple vars on RHS when not observed
        #     break
        # Alternative way of making sure algo doesn't break when all lhs variables observed
        tol = (ll_old - ll) / min([ll_old, -1])
        ll_old = ll
        it += 1
        print(f"Log Likelihood: {ll}, Tolerance: {tol}, Iteration: {it}")
    x_fit = np.zeros(T)
    p_fit = np.zeros(T)
    if weights is not None:
        x = x / weights
        for j in range(k):
            Y[:, j] /= weights
    for t in range(T):
        y_obs = Y[t, :]
        is_finite = np.isfinite(y_obs)
        is_finite[0] = False  # don't look at true val to get fitted val
        y_obs = y_obs[is_finite]
        b_obs = b[is_finite]
        h_obs = h[is_finite]
        r_obs = r[is_finite]
        q_obs = q[is_finite]
        if np.all(~is_finite):  # if nothing is observed
            x_fit[t] = np.nan
        else:
            xt = b_obs[0] * y_obs[0]  # prior (based on observation of series 0)
            pt = q_obs[0]  # prior variance
            for j in range(1, len(y_obs)):
                s = h_obs[j] * pt * h_obs[j] + r_obs[j]  # update variance
                c = h_obs[j] * pt  # update covariance
                nu = y_obs[j] - h_obs[j] * xt  # prediction error
                xt += (c / s) * nu  # update based on observation of series j
                pt -= (c**2) / s  # new variance of updated xt
            x_fit[t] = xt
            p_fit[t] = pt
    x[~np.isfinite(Y[:, 0])] = x_fit[~np.isfinite(Y[:, 0])]  # update to the latest parameters
    # if scale_input:
    #     x = x * scale[0] + center[0]
    #     x_fit = x_fit * scale[0] + center[0]
    return dict(final=x, fitted=x_fit, var_fit=p_fit, p=p, h=h, b=b, r=r, q=q)  # x, x_fit

def pool_fcast(df, lhs=None):
    """
    Calculate optimal blended forecast ignoring forecast correlations (this prevents overfitting due to
    multicollinearity. Otherwise, you could just run OLS on your forecasts. Don't do that).
    Args:
        df: dataframe with target series (LHS variable) ordered first if lhs not specified.


    Returns:

    """
    if lhs is None:
        beta_in = np.ones(df.shape[1] - 1)
    else:
        beta_in = np.ones(df.shape[1])
    return fit_series(df, lhs=lhs, beta_in=beta_in)