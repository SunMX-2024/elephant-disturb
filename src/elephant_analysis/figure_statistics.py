"""Numerical summaries retained from the original Fig. 1 plotting workflow."""
import numpy as np
from scipy import stats
from .array_utils import nanravel


def summarize_vegetation_contrasts(paired: dict) -> dict:
    result = {}
    for index, label in enumerate(("th", "tc")):
        values = np.concatenate([nanravel(paired["delta_protect_250613"][index]),
                                 nanravel(paired["delta_unprotect_250613"][index])])
        mean = values.mean()
        standard_error = stats.sem(values)
        result[label + "_values"] = values
        result[label + "_mean"] = np.asarray(mean)
        result[label + "_se"] = np.asarray(standard_error)
        result[label + "_ci"] = np.asarray(stats.t.interval(0.95, len(values) - 1, loc=mean, scale=standard_error))
        # Retain the source's numerical statistic, but name it correctly.
        # A KS normality p-value is not a test of an elephant-associated contrast.
        result[label + "_ks_normal_p"] = np.asarray(stats.kstest(values, "norm").pvalue)
        result[label + "_n"] = np.asarray(len(values))
        x = np.linspace(-15, 15, 1000)
        kde = stats.gaussian_kde(values)
        result[label + "_kde_x"] = x
        result[label + "_kde_y"] = kde(x)
        result[label + "_kde_mean"] = kde(mean)
    return result


def prepare_temperature_plot_data(paired_arrays: dict) -> dict:
    """Preserve source masks/order and return portable figure inputs."""
    delta_veg_protec = paired_arrays["delta_protect_250613"]
    delta_veg_unprotec = paired_arrays["delta_unprotect_250613"]
    delta_lst1214_protec = paired_arrays["delta_lst1214_protect_250613"]
    delta_lst1214_unprotec = paired_arrays["delta_lst1214_unprotect_250613"]

    delta_lst1516 = np.append(nanravel(delta_veg_protec[2]), nanravel(delta_veg_unprotec[2]))
    delta_lst1214 = np.append(nanravel(delta_lst1214_protec), nanravel(delta_lst1214_unprotec))
    df_delta = [delta_lst1516, delta_lst1214]

    # Common valid-cell support for TH, TC, and 2015-16 LST, by area stratum.
    delta_dd_protec = delta_veg_protec[:3].copy()
    dd = delta_dd_protec.copy()
    dd[~np.isnan(dd)] = 1
    cc_mask_protec = np.ones(delta_dd_protec.shape[1:], dtype=float)
    for i in range(3):
        cc_mask_protec *= dd[i]
    for i in range(3):
        delta_dd_protec[i] *= cc_mask_protec

    delta_dd_unprotec = delta_veg_unprotec[:3].copy()
    dd = delta_dd_unprotec.copy()
    dd[~np.isnan(dd)] = 1
    cc_mask_unprotec = np.ones(delta_dd_unprotec.shape[1:], dtype=float)
    for i in range(3):
        cc_mask_unprotec *= dd[i]
    for i in range(3):
        delta_dd_unprotec[i] *= cc_mask_unprotec

    pd_delta = []
    for i in range(3):
        pd_delta.append(np.append(nanravel(delta_dd_protec[i]), nanravel(delta_dd_unprotec[i])))
    del_tree = [pd_delta[0], pd_delta[1]]
    del_lst = pd_delta[2]
    result = {
        'lst_1516': df_delta[0],
        'lst_1214': df_delta[1],
        'th': del_tree[0],
        'tc': del_tree[1],
        'lst': del_lst,
    }
    import statsmodels.api as sm
    result['distribution_ks_p'] = np.array([stats.kstest(v, 'norm').pvalue for v in df_delta])
    for index, label in enumerate(('th', 'tc')):
        x = np.asarray(del_tree[index])
        model = sm.OLS(del_lst, sm.add_constant(x.reshape(-1, 1))).fit()
        grid = np.arange(-10, 20, dtype=float) if index == 0 else np.arange(-50, 75, dtype=float)
        prediction = model.get_prediction(sm.add_constant(grid.reshape(-1, 1)))
        result[label + '_ols_slope'] = np.asarray(model.params[1])
        result[label + '_ols_slope_ci'] = np.asarray(model.conf_int(alpha=0.05)[1])
        result[label + '_ols_p'] = np.asarray(model.pvalues[1])
        result[label + '_r2'] = np.asarray(np.corrcoef(x, del_lst)[0, 1] ** 2)
        result[label + '_fit_x'] = grid
        result[label + '_fit_y'] = prediction.predicted_mean
        result[label + '_fit_ci'] = prediction.conf_int(alpha=0.05)
    return result


def prepare_resistance_plot_data(paired_arrays: dict) -> dict:
    """Preserve source masks/order and return portable figure inputs."""
    delta_res_protec1 = paired_arrays["delta_res_protect_250613"]
    delta_res_unprotec1 = paired_arrays["delta_res_unprotect_250613"]
    delta_res_reduc_protec1 = paired_arrays["delta_res_reduc_protect_250613"]
    delta_res_reduc_unprotec1 = paired_arrays["delta_res_reduc_unprotect_250613"]
    mm_protec = paired_arrays["mask_reduc_protect_250613"]
    mm_unprotec = paired_arrays["mask_reduc_unprotect_250613"]
    dd_mask_protec = paired_arrays["geo_mask_protected"]
    dd_mask_non_protec = paired_arrays["geo_mask_non_protected"]

    res_name = ["ndvi_res", "csif_res"]
    d_DAA_protec = [
        paired_arrays[f"Vegetation_structure_protected_DAA_{name}_0.25deg_25613"] * dd_mask_protec
        for name in res_name
    ]
    d_DAA_unprotec = [
        paired_arrays[f"Vegetation_structure_non-protected_DAA_{name}_0.25deg_25613"] * dd_mask_non_protec
        for name in res_name
    ]

    relative_delta_protec = []
    relative_delta_unprotec = []
    for i in range(2):
        pre = (delta_res_protec1[i] / (1 - d_DAA_protec[i])) * 100
        values = nanravel(pre * mm_protec[i])
        relative_delta_protec.append(values[(values >= -1000) & (values <= 1000)])
        pre = (delta_res_unprotec1[i] / (1 - d_DAA_unprotec[i])) * 100
        values = nanravel(pre * mm_unprotec[i])
        relative_delta_unprotec.append(values[(values >= -1000) & (values <= 1000)])

    # Preserve the archived reduction means across protected and non-protected cells.
    delta_ndvi = np.nanmean(np.stack((delta_res_reduc_protec1[0], delta_res_reduc_unprotec1[0])), axis=0)
    delta_sif = np.nanmean(np.stack((delta_res_reduc_protec1[1], delta_res_reduc_unprotec1[1])), axis=0)
    delta = [delta_ndvi, delta_sif]
    res_delta = [
        np.append(nanravel(delta_res_reduc_protec1[i]), nanravel(delta_res_reduc_unprotec1[i]))
        for i in range(2)
    ]
    relative_delta = [
        np.append(relative_delta_protec[i], relative_delta_unprotec[i])
        for i in range(2)
    ]

    result = {
        'map_ndvi': delta[0],
        'map_sif': delta[1],
        'delta_ndvi': res_delta[0],
        'delta_sif': res_delta[1],
        'relative_ndvi': relative_delta[0],
        'relative_sif': relative_delta[1],
    }
    result['delta_ks_p'] = np.array([stats.kstest(v, 'norm').pvalue for v in res_delta])
    result['relative_ks_p'] = np.array([stats.kstest(v, 'norm').pvalue for v in relative_delta])
    result['delta_se'] = np.array([stats.sem(v) for v in res_delta])
    result['relative_se'] = np.array([stats.sem(v) for v in relative_delta])
    return result
