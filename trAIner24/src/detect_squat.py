from flask import Flask, render_template, Response
from imutils.video import VideoStream
import cv2
import time
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
from s5_test import MultiPersonClassifier

import datetime

if True:  # Include project path
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

    import utils.lib_images_io as lib_images_io
    import utils.lib_plot as lib_plot
    import utils.lib_commons as lib_commons
    from utils.lib_openpose import SkeletonDetector
    from utils.lib_tracker import Tracker
    from utils.lib_tracker import Tracker
    from utils.lib_classifier import ClassifierOnlineTest
    from utils.lib_classifier import *


#¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶
def squatsoofs(frame, pumodel):
    (h, w) = frame.shape[:2]
    img = cv2.resize(frame, (432,368))
    img = img_to_array(img)
    img = img.reshape(-1,432,368,3)
    prediction = pumodel.predict(img)
    return prediction[0]
#¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶



###===###===###===###===###===###===###===###===###===###===###===###===###===###===###===###===###===###===###===###===###===###===
def gen(maxcount, folder_name):

    #ç¢ç¢ç¢ç¢ç¢ç¢ç¢ç¢ç¢ç¢ç¢ç¢ç¢ç¢ç¢ç¢ç¢ç¢ç¢ç¢ç¢
    timer=0
    timer_countdown = int(3)
    
    a=0
    pu_list=[]
    down=0
    up=0
    no_action=0
    count=0
    correct_count=0
    incorrect_count=0
    hyper_count=0
    start_to_save=0
    folder_num=1
    img_num=1
    DST_VIDEO_NAME = "rawaction_squats_"+str(folder_num) +".mp4"
    #DST_VIDEO_NAME = "action_squats_"+str(folder_num) +".mp4"

    ##########################################
    #global input_folder_name
    output_folder_name = folder_name

    #output_folder_name = output_folder_name+"/action_squats_"+str(folder_num)
        
    DST_FOLDER = "src/static/" + output_folder_name + "/"
    
    #os.makedirs(DST_FOLDER, exist_ok=True)
    
   
    SRC_DATA_TYPE = "folder"

    SRC_MODEL_PATH = "model/trained_classifier_squat_1223.pickle"

    cfg_all = lib_commons.read_yaml(ROOT + "config/config.yaml")
    cfg = cfg_all["s5_test.py"]

    #CLASSES = np.array(cfg_all["classes"])
    CLASSES = np.array(['correct','incorrect'])
    SKELETON_FILENAME_FORMAT = cfg_all["skeleton_filename_format"]

    # Action recognition: number of frames used to extract features.
    WINDOW_SIZE = int(cfg_all["features"]["window_size"])

    # Output folder
    #DST_FOLDER = args.output_folder + "/" + DST_FOLDER_NAME + "/"
    DST_SKELETON_FOLDER_NAME = cfg["output"]["skeleton_folder_name"]
    #DST_VIDEO_NAME = cfg["output"]["video_name"]
    # framerate of output video.avi
    #DST_VIDEO_FPS = float(cfg["output"]["video_fps"])
    DST_VIDEO_FPS =6.2

    # Video setttings

    # If data_type is webcam, set the max frame rate.
    SRC_WEBCAM_MAX_FPS = float(cfg["settings"]["source"]
                            ["webcam_max_framerate"])

    # If data_type is video, set the sampling interval.
    # For example, if it's 3, then the video will be read 3 times faster.
    SRC_VIDEO_SAMPLE_INTERVAL = int(cfg["settings"]["source"]
                                    ["video_sample_interval"])

    # Openpose settings
    OPENPOSE_MODEL = cfg["settings"]["openpose"]["model"]
    OPENPOSE_IMG_SIZE = cfg["settings"]["openpose"]["img_size"]

    # Display settings
    img_disp_desired_rows = int(cfg["settings"]["display"]["desired_rows"])


    #####################################################
    #functions for openpose
    def select_images_loader(src_data_type, src_data_path):
        if src_data_type == "video":
            images_loader = lib_images_io.ReadFromVideo(
                src_data_path,
                sample_interval=SRC_VIDEO_SAMPLE_INTERVAL)

        elif src_data_type == "folder":
            images_loader = lib_images_io.ReadFromFolder(
                folder_path=src_data_path)

        elif src_data_type == "webcam":
            if src_data_path == "":
                webcam_idx = 0
            elif src_data_path.isdigit():
                webcam_idx = int(src_data_path)
            else:
                webcam_idx = src_data_path
            images_loader = lib_images_io.ReadFromWebcam(
                SRC_WEBCAM_MAX_FPS, webcam_idx)
        return images_loader

    def remove_skeletons_with_few_joints(skeletons):
        ''' Remove bad skeletons before sending to the tracker '''
        good_skeletons = []
        for skeleton in skeletons:
            px = skeleton[2:2+13*2:2]
            py = skeleton[3:2+13*2:2]
            num_valid_joints = len([x for x in px if x != 0])
            num_leg_joints = len([x for x in px[-6:] if x != 0])
            total_size = max(py) - min(py)
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # IF JOINTS ARE MISSING, TRY CHANGING THESE VALUES:
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if num_valid_joints >= 5 and total_size >= 0.1 and num_leg_joints >= 0:
                # add this skeleton only when all requirements are satisfied
                good_skeletons.append(skeleton)
        return good_skeletons


    def draw_result_img(img_disp, img_num, humans, dict_id2skeleton,
                        skeleton_detector, multiperson_classifier):
        ''' Draw skeletons, labels, and prediction scores onto image for display '''

        # Resize to a proper size for display
        r, c = img_disp.shape[0:2]
        desired_cols = int(1.0 * c * (img_disp_desired_rows / r))
        img_disp = cv2.resize(img_disp,
                            dsize=(desired_cols, img_disp_desired_rows))

        # Draw all people's skeleton
        skeleton_detector.draw(img_disp, humans)

        # Draw bounding box and label of each person
        if len(dict_id2skeleton):
            for id, label in dict_id2label.items():
                skeleton = dict_id2skeleton[id]
                # scale the y data back to original
                skeleton[1::2] = skeleton[1::2] / scale_h
                # print("Drawing skeleton: ", dict_id2skeleton[id], "with label:", label, ".")
                lib_plot.draw_action_result(img_disp, id, skeleton, label)

        # Add blank to the left for displaying prediction scores of each class
        img_disp = lib_plot.add_white_region_to_left_of_image(img_disp)

        cv2.putText(img_disp, "Frame:" + str(img_num),
                    (20, 20), fontScale=1.5, fontFace=cv2.FONT_HERSHEY_PLAIN,
                    color=(0, 0, 0), thickness=2)

        # Draw predicting score for only 1 person
        if len(dict_id2skeleton):
            classifier_of_a_person = multiperson_classifier.get_classifier(
                id='min')
            classifier_of_a_person.draw_scores_onto_image(img_disp)
        return img_disp


    def get_the_skeleton_data_to_save_to_disk(dict_id2skeleton):
        '''
        In each image, for each skeleton, save the:
            human_id, label, and the skeleton positions of length 18*2.
        So the total length per row is 2+36=38
        '''
        skels_to_save = []
        for human_id in dict_id2skeleton.keys():
            label = dict_id2label[human_id]
            skeleton = dict_id2skeleton[human_id]
            skels_to_save.append([[human_id, label] + skeleton.tolist()])
        return skels_to_save



    #ç¢ç¢ç¢ç¢ç¢ç¢ç¢ç¢ç¢ç¢ç¢ç¢ç¢ç¢ç¢ç¢ç¢ç¢ç¢ç¢ç¢
    
    cap = VideoStream(src=0).start()
    #≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈first frame (optical flow)≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈
    first_frame = cap.read()
    first_frame = cv2.resize(first_frame,(432,368))
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(first_frame)
    mask[..., 1] = 255
    #≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈first frame (optical flow)≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈
    
    pumodel = load_model(CURR_PATH+"squats_beta2.model")

    #variables for openpose
    label_list=[]
    skeleton_detector = SkeletonDetector(OPENPOSE_MODEL, OPENPOSE_IMG_SIZE)

    multiperson_tracker = Tracker()

    multiperson_classifier = MultiPersonClassifier(SRC_MODEL_PATH, CLASSES)

    #global result_list
    result_list = []
    
    while True:
        img = cap.read()
        img = cv2.resize(img, (1400, 800))
        
        #__________________stage 2 (timing&counting)________________________________________________________________
        if timer_countdown < 0:
        
            frame = cv2.resize(img,(432,368))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mask[..., 0] = angle * 180 / np.pi / 2
            mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
            
            if a%1==0:
                prediction = squatsoofs(rgb, pumodel)
                if prediction[1]==1:
                    down+=1
                elif prediction[2]==1:
                    up+=1
                else:
                    no_action+=1
                
                if down>1:
                    pu_list.append(1)
                    down=0
                    up=0
                    start_to_save=1
                elif up>1:
                    pu_list.append(2)
                    down=0
                    up=0
                elif no_action>=30:
                    pu_list=[]
                    no_action=0
                    
                #∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ  save  ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ
                if (no_action>2) & (count!=hyper_count) & (start_to_save==1):           #action seperateion
                    folder_num+=1
                    DST_VIDEO_NAME = "rawaction_squats_"+str(folder_num) +".mp4"
                    video_writer = lib_images_io.VideoWriter(DST_FOLDER + DST_VIDEO_NAME, DST_VIDEO_FPS)
                    #os.system("ffmpeg -i Video.mp4 -vcodec libx264 Video2.mp4")
                    hyper_count+=1
                    start_to_save=0
                    img_num=1
                elif start_to_save==1:                                               #save image
                    if folder_num==1:    #make first video
                        if not os.path.exists(DST_FOLDER):
                            os.makedirs(DST_FOLDER)
                            video_writer = lib_images_io.VideoWriter(DST_FOLDER + DST_VIDEO_NAME, DST_VIDEO_FPS)
                    #cv2.imwrite("action_squats_"+str(folder_num)+"/"+str(img_num)+".jpg"  ,  frame )
                    img_num+=1

                    ### code for openpose
                    img_disp = frame.copy()
                    print(f"\nProcessing image {img_num} ...")

                    # -- Detect skeletons
                    humans = skeleton_detector.detect(frame)
                    skeletons, scale_h = skeleton_detector.humans_to_skels_list(humans)
                    #skeletons = remove_skeletons_with_few_joints(skeletons)

                    # -- Track people
                    dict_id2skeleton = multiperson_tracker.track(
                        skeletons)  # int id -> np.array() skeleton

                    # -- Recognize action of each person
                    if len(dict_id2skeleton):
                        dict_id2label = multiperson_classifier.classify(
                            dict_id2skeleton)

                    # -- Draw
                    img_disp = draw_result_img(img_disp, img_num, humans, dict_id2skeleton,
                                        skeleton_detector, multiperson_classifier)

                    # Print label of a person
                    if len(dict_id2skeleton):
                        min_id = min(dict_id2skeleton.keys())
                        print("prediced label is :", dict_id2label[min_id])
                        
                        label_list.append(dict_id2label[min_id])

                    

                    video_writer.write(img_disp)

                #∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ  save  ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ∫ƒ
                    
                #[1,1,1,2,2,2,2,2,1,1,1,1,1,2,1,2,2,2,2,2,2]
                try:
                    if (pu_list[0:3]==[1,1,1]) & (pu_list[-3:]==[2,2,2]) :
                        video_writer.stop()
                        
                        count+=1
                        pu_list =[]
                        no_action=0
                        outcome = max(label_list,key=label_list.count)
                        if outcome == "correct":
                            print("Excellent!")
                            result_list.append("Correct")
                            correct_count+=1
                        elif outcome == "incorrect":
                            print("Wrong!")
                            result_list.append("Incorrect")
                            incorrect_count+=1
                        label_list=[]
                        
                    elif (pu_list[0]==2):
                        pu_list=[]
                    elif (pu_list[0]==1) & (pu_list[1]==2) :
                        pu_list=[]
                        
                except Exception as e:
                    pass
            prev_gray = gray
            a+=1
            print(pu_list)
            
            #∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞  timing   ∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞
            cur_time=time.time()
            timer = cur_time - pre_time
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(round(timer,2)),
                        (100, 780), font,
                        2, (100, 100, 255),
                        4, cv2.LINE_AA)
            #∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞  timing   ∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞
            
            cv2.putText(img, "Total Count: "+ str(count),
                    (800,660),font,
                    2, (0,255,255),
                    4, cv2.LINE_AA)
            cv2.putText(img, "Correct Count: "+ str(correct_count),
                    (800,720),font,
                    2, (0,255,0),
                    4, cv2.LINE_AA)
            cv2.putText(img, "Incorrect Count: "+ str(incorrect_count),
                    (800,780),font,
                    2, (0,0,255),
                    4, cv2.LINE_AA)
            
            
                        
        #------------------stage 2 (timing&counting)----------------------------------------------------------------
        previous = time.time()
        #____________________stage 1 (countdown)____________________________________________________________________
        while timer_countdown >= 0:
            img = cap.read()
            img = cv2.resize(img, (1400, 800))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(timer_countdown),
                        (650, 450), font,
                        6, (200, 0, 255),
                        10, cv2.LINE_AA)
            ret, jpeg = cv2.imencode('.jpg', img)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            current = time.time()
            if current-previous >= 1:
                previous = current
                timer_countdown = timer_countdown-1
            pre_time=time.time()
        #---------------------stage 1 (countdown)-------------------------------------------------------------------

        






        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        if count == maxcount:
            cv2.putText(img, "FINISHED!",
                (100,450),font,
                8, (0,0,255),
                10, cv2.LINE_AA)
            ret, jpeg = cv2.imencode('.jpg', img)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

            with open(DST_FOLDER+output_folder_name+'.txt','w') as f:
                for i in result_list:
                    f.write(i+'\n')
            for i in range(1,count+1):
                ORIGINAL_VIDEO_NAME = "rawaction_squats_"+str(i) +".mp4"
                MOD_VIDEO_NAME = "action_squats_"+str(i) +".mp4"
                #DST_VIDEO_NAME_MOD = "modaction_squats_"+str(folder_num) +".mp4"
                os.system(f"ffmpeg -i {DST_FOLDER + ORIGINAL_VIDEO_NAME} -vcodec libx264 {DST_FOLDER + MOD_VIDEO_NAME}")
            break
###===###===###===###===###===###===###===###===###===###===###===###===###===###===###===###===###===###===###===###===###===###===



