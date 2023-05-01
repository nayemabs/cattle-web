
import numpy as np
import random
import joblib
from PIL import Image
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                        process_mmdet_results)
from mmdet.apis import inference_detector, init_detector
from mmseg.apis import init_segmentor, inference_segmentor
import os
import math
import torch, torchvision


def adjust(segImg):
    im_width = segImg.width
    im_height = segImg.height
    upper_px =()
    lower_px = ()
    for i in range(im_height):
        flag=0
        for j in range(im_width):
            coordinate = j,i
            if segImg.getpixel(coordinate) == 1:
                upper_px = (j,i)
                flag = 1
                break
        
        if flag== 1:
            break
    for i in reversed(range(im_height)):
        flag=0
        for j in range(im_width):
            coordinate = j,i
            if segImg.getpixel(coordinate) == 1:
                lower_px= (j,i)
                flag = 1
                break
        
        if flag== 1:
            break
    return upper_px,lower_px

def y_distancae(point1 , point2):
    return round(abs(point1[1]-point2[1]))

    # pass
def predict(side_fname,rear_fname):


    print(torch.__version__, torch.cuda.is_available())


        

    try:
        print("Try stage")
        seg_config_file = 'models/v1/seg/deeplabv3plus_r101-d8_512x512_40k_voc12aug.py'
        seg_checkpoint_file = 'models/v1/seg/iter_40000.pth'


        rear_pose_config = 'models/v1/rear_pose/res152_animalpose_256x256.py'
        rear_pose_checkpoint = 'models/v1/rear_pose/epoch_210.pth'
        side_pose_config = 'models/v1/side_pose/res152_animalpose_256x256.py'
        side_pose_checkpoint = 'models/v1/side_pose/epoch_210.pth'
        det_config = 'models/v1/det/faster_rcnn_r50_fpn_coco.py'
        # det_config = 'models/v1/det/yolox_x_8x8_300e_coco.py'
        det_checkpoint = 'models/v1/det/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
        # det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
        weight_filename = "models/v1/weight_joblib/model_v3.joblib"

        # initialize seg & pose model
        # build the model from a config file and a checkpoint file
        model = init_segmentor(seg_config_file, seg_checkpoint_file, device='cpu')

        rear_pose_model = init_pose_model(rear_pose_config, rear_pose_checkpoint)
        side_pose_model = init_pose_model(side_pose_config, side_pose_checkpoint)
     
        print("Load pose model")
        # initialize detector
        # rear_det_model = init_detector(det_config, det_checkpoint)
        try:
            side_det_model = init_detector(det_config, det_checkpoint)
            print(type(side_det_model))
        except:
            print('something really fucked up')
            
        print("Load det model")
        # load weight predict file

        loaded_model = joblib.load(weight_filename)
        print("Load weight model")
        # model=model
        # rear_pose_model=rear_pose_model
        # side_pose_model=side_pose_model
        # side_det_model=side_det_model
        # loaded_model=loaded_model
        rear_img = rear_fname
        side_img = side_fname

        # inference detection
        rear_mmdet_results = inference_detector(side_det_model, rear_fname)
        side_mmdet_results = inference_detector(side_det_model, side_fname)
        
        # print(len(side_mmdet_results))
        # print(len(rear_mmdet_results))
        # extract person (COCO_ID=1) bounding boxes from the detection results
        rear_person_results = process_mmdet_results(rear_mmdet_results, cat_id=20)
        side_person_results = process_mmdet_results(side_mmdet_results, cat_id=20)
        # print(side_person_results)
        # print(f'side objcts:{len(side_person_results)}')
        # print(f'rear objcts:{len(rear_person_results)}')

        side_seg_result = inference_segmentor(model, side_img)
        rear_seg_result = inference_segmentor(model, rear_img)

        # print(f' seg-res {type(side_seg_result)}')
        # print(side_seg_result)
        seg = np.asarray(side_seg_result)
        sticker = cattle = bg = 0

        sticker = (seg == 2).sum()

        cattle = (seg == 1).sum()
        print("smooth till here")


        # bg = (seg == 0).sum()

        # diameter = np.sqrt(sticker) / 3.1416
        # print(f'Diameter = {diameter}')
        # print(f'sticker={sticker}')
        # print(f'cattle={cattle}')
        # print(bg)
        # print(f'uniques: {np.unique(seg)}')
        #if len(side_person_results)<1:
        #    predicted_cattle_weight = 0
        #    status = "Side image doesn't have a cattle."
        #    res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
        #    #os.remove(side_img)
        #    #os.remove(rear_img)
        #    return res
        #elif len(side_person_results)>10:
        #    predicted_cattle_weight = 0
        #    status = "Please change the side image."
        #    res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
        #    #os.remove(side_img)
        #    #os.remove(rear_img)
        #    return res       
        #elif len(rear_person_results)<1:
        #    predicted_cattle_weight = 0
        #    status = "Rear image doesn't have a cattle."
        #    res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
        #    #os.remove(side_img)
        #    #os.remove(rear_img)
        #    return res 
        #elif len(rear_person_results)>10:
        #    predicted_cattle_weight = 0
        #    status = "Please change the rear image."
        #    res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
        #    #os.remove(side_img)
        #   #os.remove(rear_img)
        #    return res         # inference Segmentation
        if sticker<100:
            predicted_cattle_weight = 0
            status = "Please apply sticker correctly."
            res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
            #os.remove(side_img)
            #os.remove(rear_img)
            return res 
        # inference pose
        rear_pose_results, rear_returned_outputs = inference_top_down_pose_model(rear_pose_model,
                                                                                rear_img,
                                                                                rear_person_results,
                                                                                bbox_thr=0.3,
                                                                                format='xyxy',
                                                                                dataset=rear_pose_model.cfg.data.test.type)
        side_pose_results, side_returned_outputs = inference_top_down_pose_model(side_pose_model,
                                                                                side_img,
                                                                                side_person_results,
                                                                                bbox_thr=0.3,
                                                                                format='xyxy',
                                                                            dataset=side_pose_model.cfg.data.test.type)

    # KPT rear and side
        rear_kpt = rear_pose_results[0]["keypoints"][:,0:2]
        side_kpt = side_pose_results[0]["keypoints"][:,0:2]
        # print(side_kpt)
        # print(rear_kpt)
        print(side_kpt.shape)
        print(rear_kpt.shape)
        if(side_kpt.shape!=(9,2)):
            predicted_cattle_weight = 0
            status = "please change side image."
            res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
            #os.remove(side_img)
            #os.remove(rear_img)
            return res
        if(rear_kpt.shape!=(4,2)):
            predicted_cattle_weight = 0
            status = "please change rear image."
            res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
            #os.remove(side_img)
            #os.remove(rear_img)
            return res
        rearKptID=rearx0=reary0=rearx1=reary1=rearx2=reary2=rearx3=reary3=0
        sideKptID=sidex0=sidey0=sidex1=sidey1=sidex2=sidey2=sidex3=sidey3=sidex4=sidey4=sidex5=sidey5=sidex6=sidey6=sidex7=sidey7=sidex8=sidey8=0

        for kptx,kpty in rear_kpt:
            if rearKptID == 0:
                rearx0 = kptx
                reary0 = kpty
            elif rearKptID == 1:
                rearx1 = kptx
                reary1 = kpty
            # elif rearKptID == 2:
            #     rearx2 = kptx
            #     reary2 = kpty
            # elif rearKptID == 3:
            #     rearx3 = kptx
            #     reary3 = kpty

            rearKptID+=1

        for kptx,kpty in side_kpt:

            # if sideKptID == 0:
            #     sidex0 = kptx
            #     sidey0 = kpty
            if sideKptID == 1:
                sidex1 = kptx
                sidey1 = kpty
            elif sideKptID == 2:
                sidex2 = kptx
                sidey2 = kpty
            elif sideKptID == 3:
                sidex3 = kptx
                sidey3 = kpty
            elif sideKptID == 4:
                sidex4 = kptx
                sidey4 = kpty
            # elif sideKptID == 5:
            #     sidex5 = kptx
            #     sidey5 = kpty 
            # elif sideKptID == 6:
            #     sidex6 = kptx
            #     sidey6 = kpty
            elif sideKptID == 7:
                sidex7 = kptx
                sidey7 = kpty
            elif sideKptID == 8:
                sidex8 = kptx
                sidey8 = kpty 

            sideKptID+=1




        #Crop side image from rear girth
        # print(f'seg len={len(side_seg_result)}')
        # print(f"side_seg_result : {side_seg_result[0]}")
        # print(f"side_seg_result type : {type(side_seg_result[0])}")
        # print(f"side_seg_result shape : {side_seg_result[0].shape}")
        segImg = Image.fromarray(np.array(side_seg_result[0].astype('uint8')))
        segRear = Image.fromarray(np.array(rear_seg_result[0].astype('uint8')))
        rear_p1,rear_p2 = adjust(segRear)
        rear_height = y_distancae(rear_p1,rear_p2)
        # print(Image.fromarray(side_seg_result[0].astype('uint8'), 'RGB'))
        # segImg=  (side_seg_result[0] * 255).round().astype(np.uint8)
        # print(f"segImg type: {type(segImg)}")
        # print(f"segImg : {segImg}")
        # segImg.save("tt.jpg")
        # segImg = Image.fromarray(np.uint8(side_seg_result[0]).astype('uint8'), 'RGB')
        # PIL_image = Image.fromarray(numpy_image.astype('uint8'), 'RGB')
        # print(type(segImg))

        side_im_width,side_im_height = segImg.size
        # print(f'w={side_im_width}, h={side_im_height}')
        # print(f"sidex8: {int(sidex8)} , sidex1: {int(sidex1)} ")
        if (int(sidex1)<(side_im_width/2)):
            # print(f'crop1 {0},{0},{int(sidex8)},{side_im_height}')
            seg_crop = segImg.crop((0,0,int(sidex8),side_im_height))
            side_p1,side_p2 = adjust(seg_crop)
            side_height = y_distancae(side_p1,side_p2)
            # seg_crop.save("test.jpg")
        if (int(sidex1)>(side_im_width/2)):
            # print(f'crop2 {int(sidex8)},{0},{side_im_height},{side_im_width}')
            seg_crop = segImg.crop((int(sidex8),0,side_im_width,side_im_height))
            side_p1,side_p2 = adjust(seg_crop)
            side_height = y_distancae(side_p1,side_p2)


            # seg_crop.save("test2.jpg")

        # side_Length_wither = round(((sidey1-sidey0)**2+(sidex1-sidex0)**2)**0.5)
        side_Length_shoulderbone = round(((sidey2-sidey1)**2+(sidex2-sidex1)**2)**0.5)
        side_F_Girth = round(((sidey4-sidey3)**2+(sidex4-sidex3)**2)**0.5)
        side_R_Girth = round(((sidey8-sidey7)**2+(sidex8-sidex7)**2)**0.5)
        #Depricated Height by kpts
        # side_height = round(((sidey6-sidey5)**2+(sidex6-sidex5)**2)**0.5)
        rear_width = round(((reary1-reary0)**2+(rearx1-rearx0)**2)**0.5)
        # rear_height = round(((reary3-reary2)**2+(rearx3-rearx2)**2)**0.5)
        actual_width = rear_width*(side_height/rear_height)


        # predicted_cattle_weight = loaded_model.predict(
        #     [[ side_Length_wither,	 side_Length_shoulderbone,	 side_F_Girth,	 side_R_Girth,	 side_height,
        #         rear_width,	 rear_height,	 actual_width,	 bg,	 cattle,	 sticker
        #     ]])
        # slw = side_Length_wither/ diameter
        # # sls = side_Length_shoulderbone / diameter
        # sfg = side_F_Girth / diameter
        # srg = side_R_Girth / diameter
        # # ah = side_height / diameter
        # aw = actual_width / diameter
        # print(f'cattle vs sticker ratio: {cattle/sticker}')
        predicted_cattle_weight = loaded_model.predict(
                [[ side_Length_shoulderbone,side_F_Girth,	side_R_Girth, sticker, cattle , actual_width]])
        # predicted_cattle_weight = loaded_model.predict(
        #         [[ slw,sfg,	srg, sticker, cattle , aw]])
        res: dict = {} 
        status = "ok"
        predicted_cattle_weight= float(predicted_cattle_weight)
        # status= "Please try again.Something went wrong."
        res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
        # predicted_cattle_weight= (math.exp(math.sqrt(100)))
        # res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
        #os.remove(side_img)
        #os.remove(rear_img)
        # return res
        if(cattle/sticker<50):
            predicted_cattle_weight= predicted_cattle_weight-(cattle/sticker)*0.68
            res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
            #os.remove(side_img)
            #os.remove(rear_img)
            return res
        elif(cattle/sticker<55):
            predicted_cattle_weight= predicted_cattle_weight-(cattle/sticker)*0.57
            res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
            #os.remove(side_img)
            #os.remove(rear_img)
            return res
        elif(cattle/sticker<60):
            predicted_cattle_weight= predicted_cattle_weight-(cattle/sticker)*0.48
            res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
            #os.remove(side_img)
            #os.remove(rear_img)
            return res
        elif(cattle/sticker<65):
            predicted_cattle_weight= predicted_cattle_weight-(cattle/sticker)*0.40
            res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
            #os.remove(side_img)
            #os.remove(rear_img)
            return res
        elif(cattle/sticker<67):
            predicted_cattle_weight= predicted_cattle_weight-(cattle/sticker)*0.28
            res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
            #os.remove(side_img)
            #os.remove(rear_img)
            return res
        elif(cattle/sticker<70):
            res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
            #os.remove(side_img)
            #os.remove(rear_img)
            return res
        elif(cattle/sticker<72):
            predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*0.25
            res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
            #os.remove(side_img)
            #os.remove(rear_img)
            return res
        elif(cattle/sticker<75):
            predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*0.35
            res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
            #os.remove(side_img)
            #os.remove(rear_img)
            return res
        elif(cattle/sticker<80):
            predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*0.45
            res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
            #os.remove(side_img)
            #os.remove(rear_img)
            return res
        elif(cattle/sticker<85):
            predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*0.55
            res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
            #os.remove(side_img)
            #os.remove(rear_img)
            return res
        elif(cattle/sticker<90):
            predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*0.65
            res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
            #os.remove(side_img)
            #os.remove(rear_img)
            return res
        elif(cattle/sticker<95):
            predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*0.75
            res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
            #os.remove(side_img)
            #os.remove(rear_img)
            return res
        elif(cattle/sticker<100):
            predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*0.85
            res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
            #os.remove(side_img)
            #os.remove(rear_img)
            return res
        elif(cattle/sticker<105):
            predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*0.95
            res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
            #os.remove(side_img)
            #os.remove(rear_img)
            return res
        elif(cattle/sticker<110):
            predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*1.05
            res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
            #os.remove(side_img)
            #os.remove(rear_img)
            return res
        elif(cattle/sticker<115):
            predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*1.15
            res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
            #os.remove(side_img)
            #os.remove(rear_img)
            return res
        elif(cattle/sticker<120):
            predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*1.25
            res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
            #os.remove(side_img)
            #os.remove(rear_img)
            return res
        elif(cattle/sticker<125):
            predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*1.35
            res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
            #os.remove(side_img)
            #os.remove(rear_img)
            return res
        elif(cattle/sticker<130):
            predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*1.45
            res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
            #os.remove(side_img)
            #os.remove(rear_img)
            return res
        elif(cattle/sticker<135):
            predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*1.55
            res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
            #os.remove(side_img)
            #os.remove(rear_img)
            return res
        elif(cattle/sticker<140):
            predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*1.65
            res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
            #os.remove(side_img)
            #os.remove(rear_img)
            return res

        elif(cattle/sticker<145):
            predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*1.75
            res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status} 
            #os.remove(side_img)
            #os.remove(rear_img)  
            return res 

        elif(cattle/sticker<150):
            predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*1.95
            res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
            #os.remove(side_img)
            #os.remove(rear_img)
            return res

        elif(cattle/sticker<155):
            predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*2.05
            res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
            #os.remove(side_img)
            #os.remove(rear_img)
            return res

        elif(cattle/sticker<160):
            predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*2.20
            res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
            #os.remove(side_img)
            #os.remove(rear_img)
            return res
        elif(cattle/sticker<165):
            predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*2.35
            res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
            #os.remove(side_img)
            #os.remove(rear_img)
            return res
        elif(cattle/sticker<170):
            predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*2.45
            res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
            #os.remove(side_img)
            #os.remove(rear_img)
            return res
        elif(cattle/sticker<180):
            predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*2.55
            res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
            #os.remove(side_img)
            #os.remove(rear_img)
            return res
        elif(cattle/sticker<190):
            predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*2.65
            res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
            #os.remove(side_img)
            #os.remove(rear_img)
            return res
        elif(cattle/sticker<200):
            predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*2.75
            res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
            #os.remove(side_img)
            #os.remove(rear_img)
            return res
        elif(cattle/sticker<210):
            predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*2.85
            res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
            #os.remove(side_img)
            #os.remove(rear_img)
            return res
        elif(cattle/sticker<220):
            predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*2.95
            res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
            #os.remove(side_img)
            #os.remove(rear_img)
            return res
        elif(cattle/sticker<230):
            predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*3.05
            res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
            #os.remove(side_img)
            #os.remove(rear_img)
            return res

        else:
            predicted_cattle_weight= 0
            status = "Sorry! This cattle can't be handled right now."
            res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
            #os.remove(side_img)
            #os.remove(rear_img)
            return res
        

    except:

        # predicted_cattle_weight= 0
        # status= "Please try again.Something went wrong."
        # res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
        # #os.remove(side_img)
        # #os.remove(rear_img)
        # return res

        try:
            print("except stage")
            seg_config_file = 'models/v1/seg/deeplabv3plus_r101-d8_512x512_40k_voc12aug.py'
            seg_checkpoint_file = 'models/v1/seg/iter_40000.pth'
            model = init_segmentor(seg_config_file, seg_checkpoint_file, device='cpu')
            side_seg_result = inference_segmentor(model, side_fname)
            rear_seg_result = inference_segmentor(model, rear_fname)
            # print("seg rear result ok")
            # print(f' seg-res {type(side_seg_result)}')
            # print(side_seg_result)
            # rear_seg = np.asarray(rear_seg_result)
            # rear_px = (rear_seg == 1).sum()
            seg = np.asarray(side_seg_result)
            sticker = cattle = bg = 0
        
            sticker = (seg == 2).sum()

            cattle = (seg == 1).sum()
            # print(f's = {sticker},c = {cattle},r = {rear_px}')
            status = "ok"
            predicted_cattle_weight= ((cattle+sticker)/(sticker))
            if(cattle/sticker<50):
                predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*0.60
                res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
                #os.remove(side_img)
                #os.remove(rear_img)
                return res
            elif(cattle/sticker<55):
                predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*0.65
                res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
                #os.remove(side_img)
                #os.remove(rear_img)
                return res
            elif(cattle/sticker<60):
                predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*0.68
                res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
                #os.remove(side_img)
                #os.remove(rear_img)
                return res
            elif(cattle/sticker<65):
                predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*0.70
                res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
                #os.remove(side_img)
                #os.remove(rear_img)
                return res
            elif(cattle/sticker<67):
                predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*0.74
                res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
                #os.remove(side_img)
                #os.remove(rear_img)
                return res
            elif(cattle/sticker<70):
                predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*0.78
                res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
                #os.remove(side_img)
                #os.remove(rear_img)
                return res
            elif(cattle/sticker<72):
                predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*0.82
                res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
                #os.remove(side_img)
                #os.remove(rear_img)
                return res
            elif(cattle/sticker<75):
                predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*0.87
                res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
                #os.remove(side_img)
                #os.remove(rear_img)
                return res
            elif(cattle/sticker<80):
                predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*0.91
                res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
                #os.remove(side_img)
                #os.remove(rear_img)
                return res
            elif(cattle/sticker<85):
                predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*0.95
                res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
                #os.remove(side_img)
                #os.remove(rear_img)
                return res
            elif(cattle/sticker<90):
                predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*1.00
                res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
                #os.remove(side_img)
                #os.remove(rear_img)
                return res
            elif(cattle/sticker<95):
                predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*1.10
                res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
                #os.remove(side_img)
                #os.remove(rear_img)
                return res
            elif(cattle/sticker<100):
                predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*1.15
                res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
                #os.remove(side_img)
                #os.remove(rear_img)
                return res
            elif(cattle/sticker<105):
                predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*1.25
                res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
                #os.remove(side_img)
                #os.remove(rear_img)
                return res
            elif(cattle/sticker<110):
                predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*1.30
                res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
                #os.remove(side_img)
                #os.remove(rear_img)
                return res
            elif(cattle/sticker<115):
                predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*1.30
                res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
                #os.remove(side_img)
                #os.remove(rear_img)
                return res
            elif(cattle/sticker<120):
                predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*1.40
                res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
                #os.remove(side_img)
                #os.remove(rear_img)
                return res
            elif(cattle/sticker<125):
                predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*1.50
                res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
                #os.remove(side_img)
                #os.remove(rear_img)
                return res
            elif(cattle/sticker<130):
                predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*1.60
                res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
                #os.remove(side_img)
                #os.remove(rear_img)
                return res
            elif(cattle/sticker<135):
                predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*1.70
                res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
                #os.remove(side_img)
                #os.remove(rear_img)
                return res
            elif(cattle/sticker<140):
                predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*1.80
                res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
                #os.remove(side_img)
                #os.remove(rear_img)
                return res

            elif(cattle/sticker<145):
                predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*1.90
                res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status} 
                #os.remove(side_img)
                #os.remove(rear_img)  
                return res 

            elif(cattle/sticker<150):
                predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*2.00
                res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
                #os.remove(side_img)
                #os.remove(rear_img)
                return res

            elif(cattle/sticker<155):
                predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*2.10
                res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
                #os.remove(side_img)
                #os.remove(rear_img)
                return res

            elif(cattle/sticker<160):
                predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*2.20
                res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
                #os.remove(side_img)
                #os.remove(rear_img)
                return res
            elif(cattle/sticker<165):
                predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*2.30
                res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
                #os.remove(side_img)
                #os.remove(rear_img)
                return res
            elif(cattle/sticker<170):
                predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*2.40
                res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
                #os.remove(side_img)
                #os.remove(rear_img)
                return res
            elif(cattle/sticker<180):
                predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*2.50
                res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
                #os.remove(side_img)
                #os.remove(rear_img)
                return res
            elif(cattle/sticker<190):
                predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*2.60
                res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
                #os.remove(side_img)
                #os.remove(rear_img)
                return res
            elif(cattle/sticker<200):
                predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*2.70
                res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
                #os.remove(side_img)
                #os.remove(rear_img)
                return res
            elif(cattle/sticker<210):
                predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*2.80
                res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
                #os.remove(side_img)
                #os.remove(rear_img)
                return res
            elif(cattle/sticker<220):
                predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*2.90
                res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
                #os.remove(side_img)
                #os.remove(rear_img)
                return res
            elif(cattle/sticker<230):
                predicted_cattle_weight= predicted_cattle_weight+(cattle/sticker)*3.00
                res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
                #os.remove(side_img)
                #os.remove(rear_img)
                return res

            else:
                predicted_cattle_weight= 0
                status = "Sorry! This cattle can't be handled right now."
                res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
                #os.remove(side_img)
                #os.remove(rear_img)
                return res
        except:
            predicted_cattle_weight= 0
            status= "Please try again. Cannot find a cattle."
            res = {"weight":predicted_cattle_weight,"ratio": 0 ,"remarks":status}
            # os.remove(side_img)
            # os.remove(rear_img)
            return res

        # pass
