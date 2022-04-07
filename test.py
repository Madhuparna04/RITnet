
import torch
from dataset import IrisDataset
from torch.utils.data import DataLoader 
import numpy as np
import matplotlib.pyplot as plt
from dataset import transform
import os
import os.path
from os import path
from opt import parse_args
from models import model_dict
from tqdm import tqdm
from utils import get_predictions
from dv_draw_ellipse import fit_ellipse
from dv_visualisation import draw_ellipse, draw_circle
from statistics import mean
import sys
#%%
import time

if __name__ == '__main__':
    
    args = parse_args()
   
    if args.model not in model_dict:
        print ("Model not found !!!")
        print ("valid models are:",list(model_dict.keys()))
        exit(1)

    folder_path = "D:\\fov_pipeline\\approximate-eyetracking\\ritnet\\"

    if args.useGPU:
        device=torch.device("cuda")
    else:
        device=torch.device("cpu")
        
    model = model_dict[args.model]
    model  = model.to(device)
    filename = folder_path + args.load
    if not os.path.exists(filename):
        print("model path not found !!!")
        exit(1)
        
    model.load_state_dict(torch.load(filename))
    model = model.to(device)
    model.eval()

    test_set = IrisDataset(filepath = folder_path + 'Semantic_Segmentation_Dataset/',\
                                 split = 'test',transform = transform)
    
    testloader = DataLoader(test_set, batch_size = args.bs,
                             shuffle=False, num_workers=0)

    os.makedirs('test/labels/',exist_ok=True)
    os.makedirs('test/output/',exist_ok=True)
    os.makedirs('test/mask/',exist_ok=True)
    

    with torch.no_grad():
        for i, batchdata in tqdm(enumerate(testloader),total=len(testloader)):
            img,labels,index,x,y= batchdata
            data = img.to(device)       
            output = model(data)   


            predict = get_predictions(output)

            for j in range (len(index)):       
                
                np.save('test/labels/{}.npy'.format(index[j]),predict[j].cpu().numpy())
                try:
                    plt.imsave('test/output/{}.jpg'.format(index[j]),255*labels[j].cpu().numpy())
                except:
                    pass
                
                pred_img = predict[j].cpu().numpy()/3.0
                vid_frame_shape_2d = (pred_img.shape[0] , pred_img.shape[1])
                pre_ellipse_info = fit_ellipse(pred_img)
                if pre_ellipse_info is None:
                                       
                    continue
                else:
                    (rr, cc, center, w, h, radian, ell)  = pre_ellipse_info
                    if center is not None:
                        ellipse_center_np = np.array(center)
                        pred_img = draw_ellipse(output_frame=pred_img, frame_shape=vid_frame_shape_2d, ellipse_info=pre_ellipse_info, color=[0, 1, 0])
                        pred_img = draw_circle(output_frame=pred_img, frame_shape=vid_frame_shape_2d, centre=ellipse_center_np, radius=5, color=[0, 1, 0]) 
                    foveaX = round((ellipse_center_np[0] - 200) / 2.42 * (2.42 + 5) + 1080)
                    foveaY = round((ellipse_center_np[1] - 320) / 2.42 * (2.42 + 5) + 600)
                   
                inp = img[j].squeeze() * 0.5 + 0.5
                img_orig = np.clip(inp,0,1)
                img_orig = np.array(img_orig)
                combine = np.hstack([img_orig,pred_img])
                plt.imsave('test/mask/{}.jpg'.format(index[j]),combine)
    
    os.rename('test',args.save)