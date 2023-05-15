import numpy as np 
import pandas as pd
import mediapipe as mp
import tkinter as tk
import customtkinter as ck
import pickle
from landmarks import landmarks
import cv2
from PIL import Image,ImageTk


window = tk.Tk()
window.geometry("480x700")
window.title("Training")
window.configure(bg="#eca11f")
ck.set_appearance_mode("System")

classLabel=ck.CTkLabel(window,height=40,width=120,font=("Arial",20),text_color='black',fg_color="#4fc1db")
classLabel.place(x=10,y=1)
classLabel.configure(text="STAGE")
counterLabel=ck.CTkLabel(window,height=40,width=120,font=("Arial",20),text_color='black',fg_color="#4fc1db")
counterLabel.place(x=160,y=1)
counterLabel.configure(text="REPS")
probLabel=ck.CTkLabel(window,height=40,width=120,font=("Arial",20),text_color='black',fg_color="#4fc1db")
probLabel.place(x=300,y=1)
probLabel.configure(text="PROB")
classBox=ck.CTkLabel(window,height=40,width=120,font=("Arial",20),text_color='black',fg_color="#4fc1db")
classBox.place(x=10,y=41)
classBox.configure(text="0")
counterBox=ck.CTkLabel(window,height=40,width=120,font=("Arial",20),text_color='black',fg_color="#4fc1db")
counterBox.place(x=160,y=41)
counterBox.configure(text="0")
probBox=ck.CTkLabel(window,height=40,width=120,font=("Arial",20),text_color='black',fg_color="#4fc1db")
probBox.place(x=300,y=41)
probBox.configure(text="0")

def reset_counter():
    global counter
    counter = 0 

button = ck.CTkButton(window,text="RESET",command=reset_counter,height=40,width=120,font=("Arial",20),text_color="black",fg_color="#4fc1db")
button.place(x=10,y=600)

frame = tk.Frame(height=480,width=480)
frame.place(x=10,y=90)
lmain = tk.Label(frame)
lmain.place(x=0,y=0)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(min_tracking_confidence=0.5,min_detection_confidence=0.5)

with open('deadlift.pkl','rb') as f:
    model=pickle.load(f)

cap = cv2.VideoCapture(0)
current_stage=''
counter = 0
bodylang_prob = np.array([0,0])
bodylang_class = ''

def detect():
    global current_stage
    global counter
    global bodylang_prob
    global bodylang_class

    ret,frame = cap.read()
    
    image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result = pose.process(image)
    mp_drawing.draw_landmarks(image,result.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(0,0,255),thickness=4,circle_radius=5),
                                 mp_drawing.DrawingSpec(color=(0,255,0),thickness=5,circle_radius=10),
                                 )
    try:
        row = np.array([[res.x,res.y,res.z,res.visibility] for  res in result.pose_landmarks.landmark]).flatten().tolist()
        X= pd.DataFrame([row],columns=landmarks)
        bodylang_prob = model.predict_proba(X)[0]
        bodylang_class = model.predict(X)[0]

        if bodylang_class == "down" and bodylang_prob[bodylang_prob.argmax()] > 0.7:
            current_stage = "down"
        elif current_stage == "down" and bodylang_class == 'up' and bodylang_prob[bodylang_prob.argmax()]>0.7:
            current_stage="up"
            counter +=1
            
    except Exception as e:
        print(e)

    img = image[:, :460, :]
    imgarr = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(imgarr)
    lmain.imgtk =  imgtk
    lmain.configure(image=imgtk)
    lmain.after(10,detect)

    counterBox.configure(text=counter)
    probBox.configure(text=bodylang_prob[bodylang_prob.argmax()])
    classBox.configure(text=current_stage)
detect()

window.mainloop()