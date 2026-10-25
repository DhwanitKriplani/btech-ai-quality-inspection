"""Read-only runtime/model checks; no camera opened and no inference benchmark claimed."""
import argparse,hashlib,importlib.util,json,platform,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);a=p.parse_args();root=Path(__file__).resolve().parent
    checks={'platform':platform.platform(),'architecture':platform.machine(),'python':sys.version.split()[0],'hardware_pi_tested_by_author':False,'camera_opened':False,'modules':{},'models':{}}
    for name in ['torch','torchvision','cv2','PIL','yaml','picamera2']:
        checks['modules'][name]=importlib.util.find_spec(name) is not None
    import torch,torchvision,cv2
    checks.update(torch=torch.__version__,torchvision=torchvision.__version__,opencv=cv2.__version__,cuda_available=torch.cuda.is_available(),cpu_threads=torch.get_num_threads())
    manifest=root/'model_manifest.json'
    if not manifest.exists():raise RuntimeError('Missing model_manifest.json')
    data=json.loads(manifest.read_text(encoding='utf-8'))
    for role,record in data['models'].items():
        path=root/record['file'];digest=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        checks['models'][role]={'present':path.is_file(),'hash_matches':digest==record['sha256']}
    checks['ready_for_software_demo']=all(checks['modules'][x] for x in ['torch','torchvision','cv2','PIL','yaml']) and all(v['hash_matches'] for v in checks['models'].values())
    checks['csi_requires_picamera2']=checks['modules']['picamera2']
    print(json.dumps(checks,indent=2))
    if a.output:a.output.write_text(json.dumps(checks,indent=2),encoding='utf-8')
    return 0 if checks['ready_for_software_demo'] else 1
if __name__=='__main__':raise SystemExit(main())
