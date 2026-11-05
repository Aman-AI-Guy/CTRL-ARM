"""Browser exercise of recorded mouse -> new mesh. Uses no physical camera/robot."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from playwright.sync_api import sync_playwright


def main():
    root=Path(__file__).resolve().parents[1]
    data=Path(os.getenv('TROVA_M5_DEMO_DATA') or tempfile.mkdtemp(prefix='trova-m5-browser-')).resolve()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    env={**os.environ,'TROVA_CAMERA':'demo','TROVA_ARM':'sim','TROVA_ARM_READ':'off',
         'TROVA_HARDWARE_UNLOCK':'','TROVA_MOTION_MODE':'dry_run','TROVA_MOCK':'1','XAI_API_KEY':'',
         'TROVA_DATA':str(data),'TROVA_PORT':str(port),'TROVA_HOST':'127.0.0.1','TROVA_TOKEN':''}
    process=subprocess.Popen([sys.executable,'-m','trova'],cwd=root,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    errors=[]
    try:
        for _ in range(100):
            try:
                with socket.create_connection(('127.0.0.1',port),timeout=.1):break
            except OSError:time.sleep(.1)
        with sync_playwright() as p:
            browser=p.chromium.launch(headless=True)
            page=browser.new_page(viewport={'width':1400,'height':1000})
            page.on('pageerror',lambda error:errors.append(str(error)))
            page.goto(f'http://127.0.0.1:{port}')
            page.locator('#m5panel > summary').click()
            page.get_by_role('button',name='Calibration / retained evidence',exact=True).click()
            page.wait_for_function("document.querySelector('#m5evidence').textContent.includes('Historical calibration')")
            page.get_by_role('button',name='Preview calibrated joint orbit',exact=True).click()
            page.wait_for_function("document.querySelector('#orbitpreview').textContent.includes('Select an object')")
            page.get_by_role('button',name='Import recorded mouse · 60 views',exact=True).click()
            page.wait_for_function("document.querySelector('#scanres').textContent.includes('60 views')")
            page.get_by_role('button',name='Replay retained cloud → new mesh',exact=True).click()
            page.wait_for_function("document.querySelector('#reconstructionres').textContent.includes('recorded_postprocessing_replay')",timeout=90000)
            page.wait_for_function("document.querySelectorAll('#reconstructionres img').length===4")
            assert page.locator('#reconstructionres').get_by_role('link',name='object.glb',exact=True).count()==1
            assert 'full COLMAP rerun: no' in page.locator('#reconstructionres').inner_text()
            job=page.evaluate('reconstructionJob')
            report=page.request.get(f'http://127.0.0.1:{port}/api/reconstruction/{job}').json()
            assert report['reconstructed'] and not report['textured_mesh']
            assert report['quality']['registered_images']==60
            response=page.request.get(f'http://127.0.0.1:{port}/api/reconstruction/{job}/artifacts/object.glb')
            assert response.ok and response.body()[:4]==b'glTF'
            page.get_by_role('button',name='Saved reconstruction jobs',exact=True).click()
            page.wait_for_function("document.querySelector('#reconstructionjobs button')!==null")
            page.set_viewport_size({'width':390,'height':844})
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth+1')
            assert not errors,errors
            (data/'browser_m5_result.json').write_text(json.dumps({'job':job,'data':str(data),'report':report,'errors':errors},indent=2))
            print('PASS M5 browser: 60 recorded JPEGs, new Open3D mesh, four previews, GLB download, retained provenance, mobile layout. Data: '+str(data))
            browser.close()
    finally:
        process.terminate()
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:process.kill();process.wait()

if __name__=='__main__':main()
