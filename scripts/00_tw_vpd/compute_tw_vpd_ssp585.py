#!/usr/bin/env python
import os,glob,warnings,numpy as np,xarray as xr
from tqdm import tqdm,notebook
import gc
os.environ["DASK_NUM_WORKERS"]="1";warnings.filterwarnings('ignore')
def es(t):return 6.108*np.exp((17.27*t)/(237.3+t))
def tw(tasmax,hurs):
 t=tasmax-273.15
 return(t*np.arctan(0.151977*(hurs+8.313659)**0.5)+np.arctan(t+hurs)-np.arctan(hurs-1.676331)+0.00391838*hurs**1.5*np.arctan(0.023101*hurs)-4.686035+273.15)
def process(model_id,tasmax_files,tasmin_files,hurs_files,out_dir="TW_VPD_LANDONLY_SAFE"):
 if not all([tasmax_files,tasmin_files,hurs_files]):return 0
 os.makedirs(out_dir,exist_ok=True)
 dsx=xr.open_mfdataset(sorted(tasmax_files),chunks={'time':365},parallel=False,decode_times=True,combine='by_coords')
 dsn=xr.open_mfdataset(sorted(tasmin_files),chunks={'time':365},parallel=False,decode_times=True,combine='by_coords')
 dsh=xr.open_mfdataset(sorted(hurs_files),chunks={'time':365},parallel=False,decode_times=True,combine='by_coords')
 dsx,dsn,dsh=xr.align(dsx,dsn,dsh,join='inner')
 tasmax,tasmin,hurs=dsx['tasmax'],dsn['tasmin'],dsh['hurs']
 scale=100.0 if hurs.attrs.get('units','').strip() in ['1','fraction'] else 1.0
 years=tasmax.time.dt.year.values.astype(int)
 for yr in sorted(set(years)):
  out_file=f"{out_dir}/{model_id}_TW_VPD_{yr}.nc"
  if os.path.exists(out_file) and os.path.getsize(out_file)>50000:continue
  idx=years==yr
  x=tasmax.isel(time=idx);n=tasmin.isel(time=idx);h=hurs.isel(time=idx);m=(x<1e19).isel(time=0)
  twv=tw(x,h*scale);vpd=es(x-273.15)-(h*scale/100.0)*es(n-273.15)
  twv=xr.where(m,twv,np.nan);vpd=xr.where(m,vpd,np.nan)
  ds=xr.Dataset({'tw':twv.where((twv<=x+2)&(twv>=x-15),np.nan).astype('float32'),'vpd':vpd.where((vpd>=0)&(vpd<80),np.nan).astype('float32')},coords=x.coords)
  ds.to_netcdf(out_file,encoding={'tw':{'zlib':True,'complevel':5},'vpd':{'zlib':True,'complevel':5}})
  print(f"+ {yr}")
  gc.collect()
 dsx.close();dsn.close();dsh.close()
models=[
 ("EC_Earth3_SSP585_2015-2100",glob.glob("EC_Earth3_SSP585_2015-2100/*tasmax*.nc"),glob.glob("EC_Earth3_SSP585_2015-2100/*tasmin*.nc"),glob.glob("EC_Earth3_SSP585_2015-2100/*hurs*.nc")),
 ("HadGEM3_GC31_LL_SSP585_2015_2100",glob.glob("HadGEM3-GC31-LL_ssp585_2015_2100/*tasmax*.nc"),glob.glob("HadGEM3-GC31-LL_ssp585_2015_2100/*tasmin*.nc"),glob.glob("HadGEM3-GC31-LL_ssp585_2015_2100/*hurs*.nc")),
 ("MPI-ESM1-2-HR_ssp585_2015_2100",glob.glob("MPI-ESM1-2-HR_ssp585_2015_2100/*tasmax*.nc"),glob.glob("MPI-ESM1-2-HR_ssp585_2015_2100/*tasmin*.nc"),glob.glob("MPI-ESM1-2-HR_ssp585_2015_2100/*hurs*.nc")),
]
for model_id,tmax,tmin,hurs in models:
 print(f"\n=== {model_id} ===")
 process(model_id,tmax,tmin,hurs)
print("\n✅ DONE")
