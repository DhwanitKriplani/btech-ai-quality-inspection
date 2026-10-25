"""Explicit empty-background capture, desktop or headless; operator must clear the scene."""
import argparse,json,sys,shutil
from pathlib import Path
from datetime import datetime,timezone
import cv2
from run_inspection import SourceReader

def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--camera',type=int);g.add_argument('--source',help='USB index, photo, or picamera2:0')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['rgb','nir'],default='rgb');p.add_argument('--overwrite',action='store_true');p.add_argument('--headless',action='store_true');a=p.parse_args()
    if a.camera is not None and a.camera<0:p.error('Camera index must be nonnegative')
    if a.output.exists() and not a.overwrite:raise ValueError('Reference already exists; choose a new path or explicitly use --overwrite')
    reader=SourceReader(str(a.camera) if a.camera is not None else a.source);saved=False;window='EMPTY area - SPACE save / Q cancel';frames=0
    try:
        reader.open()
        while True:
            ok,frame=reader.read()
            if not ok:raise RuntimeError('No camera/image frame received')
            frames+=1
            if a.headless:
                if frames<(1 if reader.kind=='image' else 5):continue
                key=32
            else:
                display=frame.copy();cv2.putText(display,'REMOVE objects. SPACE saves EMPTY background; Q cancels.',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
                cv2.imshow(window,display);key=cv2.waitKey(1)&255
                if key in [ord('q'),ord('Q'),27] or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:break
            if key==32:
                a.output.parent.mkdir(parents=True,exist_ok=True)
                if a.output.exists():shutil.copy2(a.output,a.output.with_name(a.output.stem+'_previous'+a.output.suffix))
                if not cv2.imwrite(str(a.output),frame):raise RuntimeError('Could not save reference')
                a.output.with_suffix('.json').write_text(json.dumps({'source':reader.value,'declared_mode':a.mode,'size_wh':[frame.shape[1],frame.shape[0]],'captured_utc':datetime.now(timezone.utc).isoformat(),'headless':a.headless,'operator_requirement':'User must provide empty inspection area, fixed camera, unchanged lighting. Source mode is not independently verified.'},indent=2));saved=True;print('Saved',a.output.resolve());break
    finally:
        reader.close()
        if not a.headless:cv2.destroyAllWindows()
    return 0 if saved else 1
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as e:print('ERROR:',e,file=sys.stderr);raise SystemExit(1)
