import numpy as np
import tkinter 
from tkinter import filedialog
from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename,asksaveasfilename
from tkinter import simpledialog
import os
import sys
import glob
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.morphology import black_tophat,disk
import scipy.ndimage as ndimage
import math
from skimage import exposure,filters
from matplotlib import pyplot as plt

from skimage.transform import hough_circle, hough_circle_peaks,hough_ellipse
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.segmentation import active_contour
from skimage.filters import gaussian
import argparse
from PIL import Image
import csv


global name
global fname
global ofname
name=""
global file
global folder
global ofolder


def openfile():
    global name
    name=""
    name = askopenfilename(initialdir="C:/",filetypes =(("JPG File", "*.jpg;*.png;*.tif"),("All Files","*.*")), title = "Input a fundus image")
    if name=="":
        error="Please try again!"
        file.set(error)
    else:
        file.set(name)
def openfolder():
    # Allow user to select a directory and store it in global variable called folder_path
    global fname
    fname=""
    if fname=="":
        error="Please try again!"
        folder.set(error)
    fname = filedialog.askdirectory()
    folder.set(fname)
    
def open_output_folder():
    # Allow user to select a directory and store it in global var called folder_path
    global ofname
    ofname=""
    if ofname=="":
        error="Please try again!"
        ofolder.set(error)
    ofname = filedialog.askdirectory()
    ofolder.set(ofname)
    

def mnf():
    global fname
    global ofname
    tempname=fname+"/*.jpg"
    path = glob.glob(tempname)
    for img in path:
        x = img.split("_")
        y=x[1]
        y=y.split(".")
        image = cv.imread(img)
        b,green_fundus,r = cv.split(image)
        clahe=cv.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        contrast_enhanced_green_fundus = clahe.apply(green_fundus)
       # Applying alternate sequential filtering (3 times closing opening)
        r1 = cv.morphologyEx(contrast_enhanced_green_fundus, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5)), iterations = 1)
        R1 = cv.morphologyEx(r1, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5)), iterations = 1)
        r2 = cv.morphologyEx(R1, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE,(11,11)), iterations = 1)
        R2 = cv.morphologyEx(r2, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE,(11,11)), iterations = 1)
        r3 = cv.morphologyEx(R2, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE,(23,23)), iterations = 1)
        R3 = cv.morphologyEx(r3, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE,(23,23)), iterations = 1)	
        f4 = cv.subtract(R3,contrast_enhanced_green_fundus)
        f5 = clahe.apply(f4)
        ret,f6 = cv.threshold(f5,15,255,cv.THRESH_BINARY)	
        mask = np.ones(f5.shape[:2], dtype="uint8") * 255	
        (contours,_) = cv.findContours(f6.copy(),cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv.contourArea(cnt) <= 200:
                cv.drawContours(mask, [cnt], -1, 0, -1)			
        im = cv.bitwise_and(f5, f5, mask=mask)
        ret,fin = cv.threshold(im,15,255,cv.THRESH_BINARY_INV)			
        newfin = cv.erode(fin, cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3)), iterations=1)	
        fundus_eroded = cv.bitwise_not(newfin)	
        xmask = np.ones(image.shape[:2], dtype="uint8") * 255
        (xcontours, _) = cv.findContours(fundus_eroded.copy(),cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)	
        for cnt in xcontours:
            shape = "unidentified"
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.04 * peri, False)   				
            if len(approx) > 4 and cv.contourArea(cnt) <= 3000 and cv.contourArea(cnt) >= 100:
                shape = "circle"	
            else:
                shape = "veins"
            if(shape=="circle"):
                cv.drawContours(xmask, [cnt], -1, 0, -1)	
	
        finimage = cv.bitwise_and(fundus_eroded,fundus_eroded,mask=xmask)	
        blood_vessels = cv.bitwise_not(finimage)
        out=ofname+"/mn"+y[0]+".jpg"
        cv.imwrite(out,blood_vessels)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    
def mn():
    global name
    global ofname
    image = cv.imread(name)
    b,green_fundus,r = cv.split(image)
    clahe=cv.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    contrast_enhanced_green_fundus = clahe.apply(green_fundus)
	# applying alternate sequential filtering (3 times closing opening)
    r1 = cv.morphologyEx(contrast_enhanced_green_fundus, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5)), iterations = 1)
    R1 = cv.morphologyEx(r1, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5)), iterations = 1)
    r2 = cv.morphologyEx(R1, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE,(11,11)), iterations = 1)
    R2 = cv.morphologyEx(r2, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE,(11,11)), iterations = 1)
    r3 = cv.morphologyEx(R2, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE,(23,23)), iterations = 1)
    R3 = cv.morphologyEx(r3, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE,(23,23)), iterations = 1)	
    f4 = cv.subtract(R3,contrast_enhanced_green_fundus)
    f5 = clahe.apply(f4)
    ret,f6 = cv.threshold(f5,15,255,cv.THRESH_BINARY)	
    mask = np.ones(f5.shape[:2], dtype="uint8") * 255	
    (contours,_) = cv.findContours(f6.copy(),cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv.contourArea(cnt) <= 200:
            cv.drawContours(mask, [cnt], -1, 0, -1)			
    im = cv.bitwise_and(f5, f5, mask=mask)
    ret,fin = cv.threshold(im,15,255,cv.THRESH_BINARY_INV)			
    newfin = cv.erode(fin, cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3)), iterations=1)	
    fundus_eroded = cv.bitwise_not(newfin)	
    xmask = np.ones(image.shape[:2], dtype="uint8") * 255
    (xcontours, _) = cv.findContours(fundus_eroded.copy(),cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)	
    for cnt in xcontours:
        shape = "unidentified"
        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.04 * peri, False)   				
        if len(approx) > 4 and cv.contourArea(cnt) <= 3000 and cv.contourArea(cnt) >= 100:
            shape = "circle"	
        else:
            shape = "veins"
        if(shape=="circle"):
            cv.drawContours(xmask, [cnt], -1, 0, -1)	
	
    finimage = cv.bitwise_and(fundus_eroded,fundus_eroded,mask=xmask)	
    blood_vessels = cv.bitwise_not(finimage)
    out=ofname+"/mn.jpg"
    cv.imwrite(out,blood_vessels)

    


def resize(img):
    width = 1024
    height = 720
    return cv.resize(img,(width,height), interpolation = cv.INTER_CUBIC)

def rgb2Red(img):
    b,g,r = cv.split(img)
    return r

"""def rgb2Green(img):
    b,g,r = cv.split(img)
    return g"""

def rgb2Gray(img):
    return cv.cvtColor(img,cv.COLOR_BGR2GRAY)

def preprocess(img):
    b,g,r = cv.split(img)
    gray = rgb2Red(img)
    gray_blur = cv.GaussianBlur(gray, (5,5), 0)
    gray = cv.addWeighted(gray, 1.5, gray_blur, -0.5, 0, gray)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(31,31))
    gray = ndimage.grey_closing(gray,structure=kernel)
    gray = cv.equalizeHist(gray)  
    return gray

def getROI(image):
    image_resized = resize(image)
    b,g,r = cv.split(image_resized)
    g = cv.GaussianBlur(g,(15,15),0)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(15,15))
    g = ndimage.grey_opening(g,structure=kernel)    
    (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(g)

    x0 = int(maxLoc[0])-110
    y0 = int(maxLoc[1])-110
    x1 = int(maxLoc[0])+110
    y1 = int(maxLoc[1])+110
    image = image_resized[y0:y1,x0:x1]
    return [image,x0,y0]

def canny(img,sigma):
    v = np.mean(img)
    sigma = sigma
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv.Canny(img, lower, upper)    
    return edged

def hough(edged,limm,limM):
    hough_radii = np.arange(limm, limM, 1)
    hough_res = hough_circle(edged, hough_radii)
    return hough_circle_peaks(hough_res, hough_radii,total_num_peaks=1)
def od():
   global name
   global ofname
   image = cv.imread(name)
   q = resize(image)
   cv.imwrite("q.jpg",q)
   roi,x0,y0 = getROI(image)
   preprocessed_roi = preprocess(roi)
   edged = canny(preprocessed_roi,0.22)
   kernel = np.ones((3,3),np.uint8)
   edged = cv.dilate(edged,kernel,iterations=3)
   accums, cx, cy, radii = hough(edged,55,80)
   cv.circle(roi, (int(cx),int(cy)), int(radii)+10, (0, 0, 0), -1)
   cv.imwrite("p.jpg",roi)
   im1=Image.open("q.jpg")
   im2=Image.open("p.jpg")
   im1.paste(im2,(x0,y0))
   #cv.imshow('images',im1)
   output=ofname+"/od.jpg"
   temp="exaudate.jpg"
   im1.save(output)
   im1.save(temp)
   cv.waitKey(0)
   cv.destroyAllWindows()
   
def odf():
    global fname
    global ofname
    tempname=fname+"/*.jpg"
    path = glob.glob(tempname)
    for img in path:
        x = img.split("_")
        y=x[1]
        y=y.split(".")
        image = cv.imread(img)
        odq="optic disc outputs/1q_"+y[0]+".jpg"
        odp="optic disc outputs/1p_"+y[0]+".jpg"
        q = resize(image)
        cv.imwrite(odq,q)
        roi,x0,y0 = getROI(image)
        preprocessed_roi = preprocess(roi)
        edged = canny(preprocessed_roi,0.22)
        kernel = np.ones((3,3),np.uint8)
        edged = cv.dilate(edged,kernel,iterations=3)
        accums, cx, cy, radii = hough(edged,55,80)
        cv.circle(roi, (int(cx),int(cy)), int(radii)+10, (0, 0, 0), -1)
        cv.imwrite(odp,roi)
        im1=Image.open(odq)
        im2=Image.open(odp)
        im1.paste(im2,(x0,y0))
        name=ofname+"/od_"+y[0]+".jpg"
        temp="ex/od_"+y[0]+".jpg"
        im1.save(name)
        im1.save(temp)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
def ed():
    global name
    global ofname
    od()
    img=cv.imread("exaudate.jpg")
    resized_image=cv.resize(img,(700,500))
    green=resized_image.copy()
    red=resized_image.copy()
    green[:,:,0]=0
    green[:,:,2]=0
    ret,thresh1 = cv.threshold(green,127,255,cv.THRESH_TOZERO_INV)
    red[:,:,0]=0
    red[:,:,1]=0
    ret,thresh2 = cv.threshold(red,230,255,cv.THRESH_TOZERO_INV)
    total=thresh1+thresh2
    output_image=ofname+"/exudates_output.jpg"
    cv.imwrite(output_image,total)
    cv.waitKey(0)
    cv.destroyAllWindows()

def edf():
    global fname
    global ofname
    counter=0
    files = glob.glob('ex/*.jpg')
    for f in files:
        os.remove(f)
        
    odf()
    for image in glob.glob("ex/*.jpg"):
        img=cv.imread(image)
        resized_image=cv.resize(img,(700,500))
        green=resized_image.copy()
        red=resized_image.copy()
        green[:,:,0]=0
        green[:,:,2]=0
        ret,thresh1 = cv.threshold(green,127,255,cv.THRESH_TOZERO_INV)
        red[:,:,0]=0
        red[:,:,1]=0
        ret,thresh2 = cv.threshold(red,230,255,cv.THRESH_TOZERO_INV)
        total=thresh1+thresh2
        output_image=ofname+"/exudates_outputs_"+str(counter)+".jpg"
        cv.imwrite(output_image,total)
        counter=counter+1
        cv.waitKey(0)
        cv.destroyAllWindows()
def dr():
    global name
    global file
    global folder
    global ofolder
    global ofname
    window=tkinter.Tk()
    window.geometry("750x750")
   # window.resizable(width=False, height=False)
    window.configure(background='#C0C0C0')
    window.title("Diabetic Retinopathy")
    titLab=Label(window,text="Diabetic Retinopathy",font=("Times New Roman",35,"bold", "underline"),bg="#C0C0C0",fg="#00008B").pack(padx=10)
    select=Button(window,text="Select File",command=openfile)
    select.place(x=400,y=80)
    file=StringVar()
    file.set("Please Select a file")
    filename=Label(window,textvariable=file,bg="#C0C0C0", font=("Times New Roman",10,"bold"))
    filename.place(x=100,y=80)
    
    select2=Button(window,text="Select Folder",command=openfolder)
    select2.place(x=400,y=130)
    folder=StringVar()
    folder.set("Please Select a folder")
    foldername=Label(window,textvariable=folder,bg="#C0C0C0", font=("Times New Roman",10,"bold"))
    foldername.place(x=100,y=130)
    
    
    select3=Button(window,text="Select Output Folder",command=open_output_folder)
    select3.place(x=400,y=180)
    ofolder=StringVar()
    ofolder.set("Please Select a folder")
    foldername=Label(window,textvariable=ofolder,bg="#C0C0C0", font=("Times New Roman",10,"bold"))
    foldername.place(x=100,y=180)
    
    micro1 = Button(window, text = "Microaneurysm for image",command=mn).place(x=10,y=230)
    micro2= Button(window, text = "Microaneurysm for folder",command=mnf).place(x=250,y=230)

    od1 = Button(window, text = "Optic Disc for image",command=od).place(x=10,y=280)
    od2= Button(window, text = "Optic Disc for folder",command=odf).place(x=250,y=280)
    
    exh1 = Button(window, text = "Exudates for image",command=ed).place(x=10,y=330)
    exh2= Button(window, text = "Exudates for folder",command=edf).place(x=250,y=330)

    window.mainloop()
    
    
if __name__=="__main__":
    dr()