from flask import Flask, render_template, request, Response, redirect, url_for
import cv2
#import speech_recognition as sr
from imutils.video import VideoStream
import time
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
#import squat_det
import json
import requests
import datetime
import collections
import ast

import detect_plank
import detect_squat

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed_plank', methods = ['GET', 'POST'])
def video_feed_plank():
    print(duration['duration'])
    return Response(detect_plank.gen(int(duration['duration']),folder_name), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_squat', methods = ['GET', 'POST'])
def video_feed_squat():
    print(count['count'])
    #print(type(count['count']))
    return Response(detect_squat.gen(int(count['count']),folder_name), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/plank', methods = ['GET', 'POST'])
def plank():
    activate=''
    global duration
    global folder_name
    if request.method == "POST"and "duration" in request.form:
        activate=1
        duration =request.form.to_dict()
        folder_name = str(datetime.datetime.now())[5:].replace(
        ' ', '-').replace(":", '-').replace('.', '-')[:-3]
    if request.method == "POST" and "result" in request.form:
        return redirect(url_for("result_plank"))
    return render_template('plank.html', activate=activate)


@app.route('/result_plank')
def result_plank():
    videofolder = os.path.join('src/static',folder_name)
    app.config['UPLOAD_FOLDER'] = videofolder
    #video1 = os.path.join(app.config['UPLOAD_FOLDER'],'action_squats_1.mp4')
    videofolder = os.path.join('src/static',folder_name)
    app.config['UPLOAD_FOLDER'] = videofolder
    #video1 = os.path.join(app.config['UPLOAD_FOLDER'],'action_squats_1.mp4')
    #videolist = os.listdir('static/'+filename)
    video = "plank.mp4"
    videopath= folder_name+"/plank.mp4"
    #videolist.sort()
    #videolist.pop()
    try:
        with open("src/static/"+folder_name+"/"+folder_name+".txt",'r') as f:
            resultdict = ast.literal_eval(f.read())
    except:
        return '<h1>You have not finished the exercise!</h1>'
    correct = resultdict['correct_percentage']
    highhip = resultdict['high_hip_percentage']
    lowhip = resultdict['low_hip_percentage']
    incorrect = round(highhip+lowhip)
    

    #videodict = {videolist[i].split(".")[0] + " : " + resultlist[i]:filename +"/" + videolist[i] for i in range(len(videolist))}
    #correctdict = [i for i in videodict.items() if i[0].split(" : ")[1] == "correct"]
    #incorrectdict = [i for i in videodict.items() if i[0].split(" : ")[1] == "incorrect"]

    return render_template('resultplank.html', video=video, videopath=videopath, correct = correct, incorrect=incorrect, highhip=highhip,lowhip=lowhip)

@app.route('/squat', methods = ['GET', 'POST'])
def squat():
    activate=''
    global count
    global folder_name
    if request.method == "POST" and "count" in request.form:
        activate=1
        count=request.form.to_dict()
        folder_name = str(datetime.datetime.now())[5:].replace(
        ' ', '-').replace(":", '-').replace('.', '-')[:-3]
    if request.method == "POST" and "result" in request.form:
        return redirect(url_for("result_squat"))

    return render_template('squat.html', activate=activate)




@app.route('/result_squat')
def result_squat():
    videofolder = os.path.join('src/static',folder_name)
    app.config['UPLOAD_FOLDER'] = videofolder
    #video1 = os.path.join(app.config['UPLOAD_FOLDER'],'action_squats_1.mp4')
    videolist = os.listdir('src/static/'+folder_name)
    
    videolist = [video for video in videolist if video.split(".")[-1]=="mp4" and video[:6]=="action"]
    videolist.sort()
    #videolist.pop()
    try:
        with open('src/static/'+folder_name+"/"+folder_name+".txt", 'r') as f:
            resultlist= [i[:-1] for i in f]
    except:
        return '<h1>You have not finished the exercise!</h1>'
    score = {'Correct': resultlist.count("Correct"), 'Incorrect': resultlist.count("Incorrect")}
    overallscore = round(score["Correct"]/(score["Correct"]+score["Incorrect"]),2)*100
    resultdict = dict(zip(videolist,resultlist))
    correctlist = [i[0] for i in resultdict.items() if i[1]=="Correct"]
    incorrectlist = [i[0] for i in resultdict.items() if i[1]=="Incorrect"]
    correctdict = {video:folder_name +"/" + video for video in correctlist}
    incorrectdict = {video:folder_name +"/" + video for video in incorrectlist}

    #videodict = {videolist[i].split(".")[0] + " : " + resultlist[i]:filename +"/" + videolist[i] for i in range(len(videolist))}
    #correctdict = [i for i in videodict.items() if i[0].split(" : ")[1] == "correct"]
    #incorrectdict = [i for i in videodict.items() if i[0].split(" : ")[1] == "incorrect"]

    return render_template('resultsquat.html', score= overallscore, score1 = score["Correct"], score2 = score["Incorrect"], 
        correctdict=correctdict, incorrectdict=incorrectdict)




if __name__=="__main__":
    app.run(debug=True)