#!/usr/bin/env python
import os, glob, warnings, numpy as np, xarray as xr
from tqdm import tqdm
import gc
os.environ["DASK_NUM_WORKERS"] = "1"
warnings.filterwarnings('ignore')

def es(t_c): return 6.108 * np.exp((17.27 * t_c) / (237.3 + t_c))
def tw_max(tasmax_k, hurs):
    t = tasmax_k - 273.15
    return (t * np.arctan(0.151977 * (hurs + 8.313659)**0.5) + np.arctan(t + hurs) - np.arctan(hurs - 1.676331) + 0.00391838 * hurs**1.5 * np.arctan(0.023101 * hurs) - 4.686035 + 273.15)

def process_ec_earth_merged(model_id, base_dir, out_dir="TW_VPD_LANDONLY_SAFE"):
    """Process EC-Earth merged files (3 files per variable covering 2015-2100)"""
    print(f"\n{'='*80}\n  MODEL: {model_id} (MERGED FILES)\n{'='*80}\n")
    
    # EC-Earth merged structure
    tasmax_files = sorted(glob.glob(f"{base_dir}/tasmax_ssp126_raw_ssp_26_nexgddp_v2/*.nc"))
    tasmin_files = sorted(glob.glob(f"{base_dir}/tasmin_ssp126_raw_ssp_26_nexgddp_v2/*.nc"))
    hurs_files = sorted(glob.glob(f"{base_dir}/hurs_ssp126_ssp26_nexgddp_v2/*.nc"))
    
    if not all([tasmax_files, tasmin_files, hurs_files]):
        print("  WARNING: missing input files")
        return 0
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Open merged files
    dsx = xr.open_mfdataset(tasmax_files, chunks={'time': 365}, parallel=False, combine='by_coords', decode_times=True)
    dsn = xr.open_mfdataset(tasmin_files, chunks={'time': 365}, parallel=False, combine='by_coords', decode_times=True)
    dsh = xr.open_mfdataset(hurs_files, chunks={'time': 365}, parallel=False, combine='by_coords', decode_times=True)
    
    dsx, dsn, dsh = xr.align(dsx, dsn, dsh, join='inner')
    tasmax, tasmin, hurs = dsx['tasmax'], dsn['tasmin'], dsh['hurs']
    
    scale = 100.0 if hurs.attrs.get('units', '').strip() in ['1', 'fraction'] else 1.0
    years = tasmax.time.dt.year.values.astype(int)
    unique_years = sorted(set(years))
    
    print(f"  Processing {len(unique_years)} years: {unique_years[0]}-{unique_years[-1]}")
    
    new = 0
    for yr_int in tqdm(unique_years, desc=model_id, unit="yr"):
        out_file = f"{out_dir}/{model_id}_TW_VPD_{yr_int}.nc"
        if os.path.exists(out_file) and os.path.getsize(out_file) > 50000:
            continue
        
        try:
            idx = years == yr_int
            x = tasmax.isel(time=idx); n = tasmin.isel(time=idx); h = hurs.isel(time=idx); m = (x < 1e19)
            tw = tw_max(x, h * scale)
            vpd = es(x - 273.15) - (h * scale / 100.0) * es(n - 273.15)
            tw = xr.where(m, tw, np.nan); vpd = xr.where(m, vpd, np.nan)
            ds = xr.Dataset({
                'tw': tw.where((tw <= x + 2) & (tw >= x - 15), np.nan).astype('float32'),
                'vpd': vpd.where((vpd >= 0) & (vpd < 80), np.nan).astype('float32')
            }, coords=x.coords)
            ds.to_netcdf(out_file, encoding={'tw': {'zlib': True, 'complevel': 5}, 'vpd': {'zlib': True, 'complevel': 5}})
            if not os.path.exists(out_file) or os.path.getsize(out_file) < 50000:
                raise RuntimeError(f"Write failed: {out_file}")
            print(f"  + {yr_int}")
            new += 1
        except Exception as e:
            print(f"  x {yr_int}: {str(e)[:100]}")
        finally:
            gc.collect()
    
    dsx.close(); dsn.close(); dsh.close()
    print(f"  DONE: {model_id} - {len(unique_years)-new} existing + {new} new")
    return new

def process_yearly(model_id, tasmax_files, tasmin_files, hurs_files, out_dir="TW_VPD_LANDONLY_SAFE"):
    """Process models with yearly files (HadGEM3, MPI-ESM)"""
    print(f"\n{'='*80}\n  MODEL: {model_id}\n  Files: tasmax={len(tasmax_files)}, tasmin={len(tasmin_files)}, hurs={len(hurs_files)}\n{'='*80}\n")
    
    if not all([tasmax_files, tasmin_files, hurs_files]):
        print("  WARNING: missing input files")
        return 0
    
    os.makedirs(out_dir, exist_ok=True)
    
    def open_safe(files):
        files = sorted(files)
        return xr.open_mfdataset(files, chunks={'time': 365}, parallel=False, combine='by_coords', decode_times=True)
    
    dsx = open_safe(tasmax_files); dsn = open_safe(tasmin_files); dsh = open_safe(hurs_files)
    dsx, dsn, dsh = xr.align(dsx, dsn, dsh, join='inner')
    tasmax, tasmin, hurs = dsx['tasmax'], dsn['tasmin'], dsh['hurs']
    
    scale = 100.0 if hurs.attrs.get('units', '').strip() in ['1', 'fraction'] else 1.0
    years = tasmax.time.dt.year.values.astype(int)
    unique_years = sorted(set(years))
    
    print(f"  Processing {len(unique_years)} years: {unique_years[0]}-{unique_years[-1]}")
    
    new = 0
    for yr_int in tqdm(unique_years, desc=model_id, unit="yr"):
        out_file = f"{out_dir}/{model_id}_TW_VPD_{yr_int}.nc"
        if os.path.exists(out_file) and os.path.getsize(out_file) > 50000:
            continue
        
        try:
            idx = years == yr_int
            x = tasmax.isel(time=idx); n = tasmin.isel(time=idx); h = hurs.isel(time=idx); m = (x < 1e19)
            tw = tw_max(x, h * scale)
            vpd = es(x - 273.15) - (h * scale / 100.0) * es(n - 273.15)
            tw = xr.where(m, tw, np.nan); vpd = xr.where(m, vpd, np.nan)
            ds = xr.Dataset({
                'tw': tw.where((tw <= x + 2) & (tw >= x - 15), np.nan).astype('float32'),
                'vpd': vpd.where((vpd >= 0) & (vpd < 80), np.nan).astype('float32')
            }, coords=x.coords)
            ds.to_netcdf(out_file, encoding={'tw': {'zlib': True, 'complevel': 5}, 'vpd': {'zlib': True, 'complevel': 5}})
            if not os.path.exists(out_file) or os.path.getsize(out_file) < 50000:
                raise RuntimeError(f"Write failed: {out_file}")
            print(f"  + {yr_int}")
            new += 1
        except Exception as e:
            print(f"  x {yr_int}: {str(e)[:100]}")
        finally:
            gc.collect()
    
    dsx.close(); dsn.close(); dsh.close()
    print(f"  DONE: {model_id} - {len(unique_years)-new} existing + {new} new")
    return new

def main():
    out_dir = "TW_VPD_LANDONLY_SAFE"
    total_new = 0
    
    # EC-Earth: merged files
    total_new += process_ec_earth_merged("EC_Earth3_SSP126_2015-2100",
                                         "/intelnvme04/haider/raw_data/nexgddp_ssp126_2015_2100/EC_Earth_ssp126_2015_2100_merged",
                                         out_dir)
    
    # HadGEM3: yearly files
    total_new += process_yearly("HadGEM3_GC31_LL_SSP126_2015_2100",
                               glob.glob("HadGEM3-GC31-LL_ssp126_2015_2100/*tasmax*.nc"),
                               glob.glob("HadGEM3-GC31-LL_ssp126_2015_2100/*tasmin*.nc"),
                               glob.glob("HadGEM3-GC31-LL_ssp126_2015_2100/*hurs*.nc"),
                               out_dir)
    
    # MPI-ESM: yearly files
    total_new += process_yearly("MPI-ESM1-2-HR_ssp126_2015_2100",
                               glob.glob("MPI-ESM1-2-HR_ssp126_2015_2100/*tasmax*.nc"),
                               glob.glob("MPI-ESM1-2-HR_ssp126_2015_2100/*tasmin*.nc"),
                               glob.glob("MPI-ESM1-2-HR_ssp126_2015_2100/*hurs*.nc"),
                               out_dir)
    
    print(f"\n{'='*80}\nJOB COMPLETE: {total_new} new files in {out_dir}\nTotal files: {len(glob.glob(f'{out_dir}/*.nc'))}\n{'='*80}")

if __name__ == "__main__":
    main()
