#!/usr/bin/env python
"""Resumable TW/VPD for NEX-GDDP historical (1981-2014) - DEADLOCK-PROOF"""
import os, glob, warnings, numpy as np, xarray as xr, pandas as pd
from tqdm import tqdm; import gc

os.environ["DASK_NUM_WORKERS"] = "1"
warnings.filterwarnings('ignore', category=UserWarning)

def es(t_c): return 6.108 * np.exp((17.27 * t_c) / (237.3 + t_c))

def tw_max(tasmax_k, hurs):
    t = tasmax_k - 273.15
    return (t * np.arctan(0.151977 * (hurs + 8.313659)**0.5) +
            np.arctan(t + hurs) - np.arctan(hurs - 1.676331) +
            0.00391838 * hurs**1.5 * np.arctan(0.023101 * hurs) -
            4.686035 + 273.15)

def process(model, tasmax_f, tasmin_f, hurs_f, out_dir="TW_VPD_LANDONLY_SAFE"):
    print(f"\n{'='*70}\n{model} | tasmax:{len(tasmax_f)} tasmin:{len(tasmin_f)} hurs:{len(hurs_f)}")
    if not all([tasmax_f, tasmin_f, hurs_f]): 
        print(f"?? SKIP {model} (missing files)"); return 0
    
    os.makedirs(out_dir, exist_ok=True)
    open = lambda f: xr.open_dataset(f[0], chunks={'time':365}, parallel=False) if len(f)==1 else xr.open_mfdataset(sorted(f), chunks={'time':365}, parallel=False, combine='by_coords')
    
    dsx, dsn, dsh = open(tasmax_f), open(tasmin_f), open(hurs_f)
    dsx, dsn, dsh = xr.align(dsx, dsn, dsh, join='inner')
    x, n, h, m = dsx['tasmax'], dsn['tasmin'], dsh['hurs'], dsx['tasmax'] < 1e19
    
    new = 0
    for yr in tqdm(range(1981,2015), desc=model, unit="yr"):
        out = f"{out_dir}/{model}_TW_VPD_{yr}.nc"
        if os.path.exists(out): continue
        
        s = slice(f"{yr}-01-01", f"{yr}-12-31")
        xyr, nyr, hyr, myr = x.sel(time=s), n.sel(time=s), h.sel(time=s), m.sel(time=s)
        if len(xyr.time) == 0: continue
        
        tw = tw_max(xyr, hyr)
        vpd = es(xyr-273.15) - (hyr/100.0)*es(nyr-273.15)
        tw, vpd = xr.where(myr, tw, np.nan), xr.where(myr, vpd, np.nan)
        
        ds = xr.Dataset({
            'tw': tw.where((tw <= xyr+2) & (tw >= xyr-15), np.nan).astype('float32'),
            'vpd': vpd.where((vpd >= 0) & (vpd < 80), np.nan).astype('float32')
        }, coords=xyr.coords)
        
        ds.to_netcdf(out, encoding={'tw':{'zlib':True,'complevel':5}, 'vpd':{'zlib':True,'complevel':5}})
        new += 1
        del xyr, nyr, hyr, myr, tw, vpd, ds; gc.collect()
    
    dsx.close(); dsn.close(); dsh.close()
    print(f"? {model}: {34-new} existing + {new} new")
    return new

def main():
    o = "TW_VPD_LANDONLY_SAFE"; total = 0
    
    # CORRECT NEX-GDDP PATTERNS (variable_day_Model_*)
    ec = ("tasmax_day_EC-Earth3_historical*.nc", "tasmin_day_EC-Earth3_historical*.nc", "hurs_day_EC-Earth3_historical*.nc")
    hg = ("tasmax_day_HadGEM3-GC31-LL_historical*.nc", "tasmin_day_HadGEM3-GC31-LL_historical*.nc", "hurs_day_HadGEM3-GC31-LL_historical*.nc")
    mpi = ("tasmax_day_MPI-ESM1-2-HR_historical*.nc", "tasmin_day_MPI-ESM1-2-HR_historical*.nc", "hurs_day_MPI-ESM1-2-HR_historical*.nc")
    
    # Fallback patterns if hyphens don't match (underscore variants)
    if not glob.glob(ec[0]): ec = ("tasmax_day_EC-Earth3_historical*.nc", "tasmin_day_EC-Earth3_historical*.nc", "hurs_day_EC-Earth3_historical*.nc")
    if not glob.glob(hg[0]): hg = ("tasmax_day_HadGEM3*_historical*.nc", "tasmin_day_HadGEM3*_historical*.nc", "hurs_day_HadGEM3*_historical*.nc")
    if not glob.glob(mpi[0]): mpi = ("tasmax_day_MPI-ESM1-2-HR_historical*.nc", "tasmin_day_MPI-ESM1-2-HR_historical*.nc", "hurs_day_MPI-ESM1-2-HR_historical*.nc")
    
    total += process("EC-Earth3", glob.glob(ec[0]), glob.glob(ec[1]), glob.glob(ec[2]), o)
    total += process("HadGEM3-GC31-LL", glob.glob(hg[0]), glob.glob(hg[1]), glob.glob(hg[2]), o)
    total += process("MPI-ESM1-2-HR", glob.glob(mpi[0]), glob.glob(mpi[1]), glob.glob(mpi[2]), o)
    
    print(f"\n{'='*70}\n? TOTAL NEW FILES: {total}\n{'='*70}")

if __name__ == "__main__": main()
