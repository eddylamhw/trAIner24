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



###===###===###===###===###===###===###===###===###===###===###===###===###===###===###===###===###===###===###===###===###===###===
def gen(duration, folder_name):

    #ç¢ç¢ç¢ç¢ç¢ç¢ç¢ç¢ç¢ç¢ç¢ç¢ç¢ç¢ç¢ç¢ç¢ç¢ç¢ç¢ç¢
    timer=0
    timer_countdown = int(3)

    total = 0
    correct = 0
    high_hip = 0
    low_hip =0

    img_num=0
    

    ##########################################
    #input_folder_name = lib_commons.get_time_string()
    output_folder_name = folder_name
    #input_folder_path = "src/data/"+ input_folder_name
    #if not os.path.exists(input_folder_path):
        #os.makedirs(input_folder_path)
    SRC_DATA_TYPE = "folder"

    SRC_MODEL_PATH = "model/trained_classifier_plank_1223.pickle"
    #SDC_DATA_PATH = "src/data/"+ input_folder_name +"/"
    DST_FOLDER = "src/static" + "/" +output_folder_name + "/"
    if not os.path.exists(DST_FOLDER):
        os.makedirs(DST_FOLDER)
    #os.makedirs(DST_FOLDER, exist_ok=True)
    DST_VIDEO_NAME = "rawplank.mp4"


    # -- Settings

    cfg_all = lib_commons.read_yaml(ROOT + "config/config.yaml")
    cfg = cfg_all["s5_test.py"]

    #CLASSES = np.array(cfg_all["classes"])
    CLASSES = np.array(['correct', 'high-hip','low-hip'])
    SKELETON_FILENAME_FORMAT = cfg_all["skeleton_filename_format"]

    # Action recognition: number of frames used to extract features.
    WINDOW_SIZE = int(cfg_all["features"]["window_size"])

    # Output folder
    #DST_FOLDER = args.output_folder + "/" + DST_FOLDER_NAME + "/"
    DST_SKELETON_FOLDER_NAME = cfg["output"]["skeleton_folder_name"]
    #DST_VIDEO_NAME = cfg["output"]["video_name"]

    # framerate of output video.avi
    #DST_VIDEO_FPS = float(cfg["output"]["video_fps"])
    DST_VIDEO_FPS = 8.7

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

    #variables for openpose

    skeleton_detector = SkeletonDetector(OPENPOSE_MODEL, OPENPOSE_IMG_SIZE)

    multiperson_tracker = Tracker()

    multiperson_classifier = MultiPersonClassifier(SRC_MODEL_PATH, CLASSES)

    #os.makedirs(DST_FOLDER, exist_ok=True)
    #os.makedirs(DST_FOLDER + DST_SKELETON_FOLDER_NAME, exist_ok=True)

    
    while True:
        img = cap.read()
        img = cv2.resize(img, (1400, 800))
        
        #__________________stage 2 (timing&counting)________________________________________________________________
        if timer_countdown < 0:
        
            frame = cv2.resize(img,(432,368))

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
                total +=1
                #'plank-correct', 'plank-incorrect-high-hip','plank-incorrect-low-hip'
                if dict_id2label[min_id] == 'correct':
                    cv2.putText(img, "Correct",(800,780), font, #might have to change
                                2, (0,255,0),
                                4, cv2.LINE_AA)
                    correct +=1
                elif dict_id2label[min_id] == 'high-hip':
                    cv2.putText(img, "High Hip Detected!",
                                (800,780), font, #might have to change
                                2, (0,255,255),
                                4, cv2.LINE_AA)
                    high_hip +=1
                elif dict_id2label[min_id] == 'low-hip':
                    cv2.putText(img, "Low Hip Detected!",
                                (800,780), font, #might have to change
                                2, (0,0,255),
                                4, cv2.LINE_AA)
                    low_hip +=1
                


            video_writer.write(img_disp)

            
            #∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞  timing   ∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞∞
            cur_time=time.time()
            timer = cur_time - pre_time
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(round(timer,2)),
                        (100, 780), font,
                        2, (100, 100, 255),
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
            video_writer = lib_images_io.VideoWriter(DST_FOLDER + DST_VIDEO_NAME, DST_VIDEO_FPS)
        #---------------------stage 1 (countdown)-------------------------------------------------------------------
            
        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        if timer >= duration:
            video_writer.stop()
            cv2.putText(img, "FINISHED!",
                (100,450),font,
                8, (0,0,255),
                10, cv2.LINE_AA)
            ret, jpeg = cv2.imencode('.jpg', img)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            total_without_no_response = correct + high_hip + low_hip
            # correct_percentage = round(correct/total*100,2)
            # high_hip_percentage = round(high_hip/total*100,2)
            # low_hip_percentage = round(low_hip/total*100,2)
            correct_percentage = round(correct/total_without_no_response*100)
            high_hip_percentage = round(high_hip/total_without_no_response*100)
            low_hip_percentage = round(low_hip/total_without_no_response*100)
            print(f"Correct Percentage: {correct_percentage}")
            result_dict = {'correct_percentage':correct_percentage, 
                            'high_hip_percentage':high_hip_percentage, 
                            'low_hip_percentage':low_hip_percentage}
            with open(DST_FOLDER+output_folder_name+'.txt','w') as f:
                f.write(str(result_dict))
            #try:
                #print(f"High Hip: {high_hip_percnetage}")
                #print(f"Low Hip: {low_hip_percnetage}")
            #except:
                #pass
            ORIGINAL_VIDEO_NAME = DST_VIDEO_NAME
            MOD_VIDEO_NAME = "plank.mp4"
                #DST_VIDEO_NAME_MOD = "modaction_squats_"+str(folder_num) +".mp4"
            os.system(f"ffmpeg -i {DST_FOLDER + ORIGINAL_VIDEO_NAME} -vcodec libx264 {DST_FOLDER + MOD_VIDEO_NAME}")
            break

###===###===###===###===###===###===###===###===###===###===###===###===###===###===###===###===###===###===###===###===###===###===





