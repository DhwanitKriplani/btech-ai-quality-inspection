"""Observed empty-background localization for a fixed inspection camera."""
from types import SimpleNamespace
import math
import cv2
import numpy as np
import torch

class BackgroundLocalizer:
    names={0:'metal',1:'plastic',2:'wood'}
    task='background_difference'
    native_resolution=True
    def __init__(self, reference, modality='rgb', threshold=25.0, min_area=.002):
        if reference.dtype!=np.uint8 or reference.ndim!=3 or reference.shape[2]!=3:
            raise ValueError('Background must be a normalized 8-bit three-channel photo')
        if not math.isfinite(threshold) or not 0<threshold<255 or not math.isfinite(min_area) or min_area<=0:
            raise ValueError('Background threshold must be in (0,255), min area positive')
        self.reference=reference.copy();self.modality=modality;self.threshold=float(threshold);self.min_area=float(min_area)
        self.component_rules={'threshold':self.threshold,'minimum_area':self.min_area,'open_kernel':3,'close_kernel':7,'confidence':'none; foreground geometry only'}
        self._reference=self._prepare(reference)
    def _prepare(self, frame):
        frame=cv2.GaussianBlur(frame,(5,5),0)
        return cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY if self.modality=='nir' else cv2.COLOR_BGR2LAB)
    def predict(self, source, max_det=12, **kwargs):
        if source.dtype!=np.uint8 or source.shape!=self.reference.shape:
            raise ValueError('Frame and empty background must have exactly the same dimensions/channels and uint8 pixels')
        difference=cv2.absdiff(self._prepare(source),self._reference)
        if difference.ndim==3:difference=difference.max(axis=2)
        mask=(difference>self.threshold).astype(np.uint8)
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
        count,_,stats,_=cv2.connectedComponentsWithStats(mask,connectivity=8)
        minimum=self.min_area*mask.size if self.min_area<1 else self.min_area
        components=[]
        for x,y,w,h,area in stats[1:]:
            if area>=minimum and w>=3 and h>=3:components.append(([int(x),int(y),int(x+w),int(y+h)],int(area)))
        components.sort(key=lambda item:item[1],reverse=True);components=components[:max_det]
        boxes=torch.tensor([b for b,_ in components],dtype=torch.float32).reshape(-1,4)
        # Zeros are interface placeholders only; the pipeline emits null localization confidence.
        fields=SimpleNamespace(xyxy=boxes,conf=torch.zeros(len(components)),cls=torch.zeros(len(components)))
        return [SimpleNamespace(boxes=fields,names=self.names,component_areas=torch.tensor([a for _,a in components]),foreground_mask=mask)]
