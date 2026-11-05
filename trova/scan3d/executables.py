"""Portable COLMAP discovery. A configured but invalid executable is not availability."""
import os
from pathlib import Path
import re
import shutil
import subprocess


def discover(value=None):
    candidate=value or os.getenv('TROVA_COLMAP') or shutil.which('colmap') or shutil.which('COLMAP.bat')
    if not candidate:return {'path':None,'available':False,'reason':'COLMAP is not installed. Set TROVA_COLMAP or add colmap to PATH.'}
    candidate=str(candidate).strip().strip('"')
    if os.name!='nt' and re.match(r'^[A-Za-z]:[\\/]',candidate):
        candidate='/mnt/'+candidate[0].lower()+'/'+candidate[3:].replace('\\','/')
    path=Path(candidate).expanduser()
    if path.is_dir():
        options=[path/'colmap',path/'COLMAP.bat',path/'colmap.exe',path/'bin/colmap']
        path=next((p for p in options if p.is_file()),path/'colmap')
    actual=shutil.which(str(path)) or (str(path.resolve()) if path.is_file() else None)
    if not actual:return {'path':None,'available':False,'reason':f'COLMAP executable does not exist: {candidate}'}
    try:
        probe=subprocess.run([actual,'-h'],capture_output=True,text=True,timeout=10)
        text=probe.stdout+probe.stderr
        if probe.returncode or not any(word in text.lower() for word in ['colmap','mapper','feature_extractor','mesh_texturer']):
            raise ValueError('The executable did not advertise COLMAP commands.')
        return {'path':actual,'available':True,'reason':'COLMAP executable validated.','help':text[:16000],
                'legacy_sift_flags':bool(re.search(r'COLMAP\s+3\.',text))}
    except (OSError,ValueError,subprocess.SubprocessError) as exc:
        return {'path':None,'available':False,'reason':f'Cannot run COLMAP: {exc}'}
