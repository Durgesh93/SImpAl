from time import time
from tensorboardX import SummaryWriter
import config as config
from tabulate import tabulate
from trainer_supervised import TrainerG
import shutil
import os
import warnings
import torch
import numpy as np
import logging
import sys
from Tkinter import *
import math
from PIL import Image, ImageDraw,ImageTk


b1 = "up"
xold, yold = None, None

root = Tk()
root.title("Image Search")
clear_btn = Button (root,text ="clear drawing",height=0)
search_btn = Button (root,text ="search images",height=0)
drawing_area = Canvas(root,height=580,width=600,bg='white')

img 		 = Image.new("RGB", (600, 600), 'white')
draw_image =  ImageDraw.Draw(img)
trainer_G = TrainerG()


def b1down(event):
	global b1
	b1 = "down" 

																																									
def search_img(event):
	global img
	img  = img.resize((config.settings['resolution'],config.settings['resolution']))
	img_arr = np.array(img)
	img_arr = torch.from_numpy(img_arr).permute(2,0,1)/255.0
	trainer_G.set_mode(config.settings['mode']['val'])
	pred_imgs = trainer_G.get_nearest_target_img(img_arr,config.settings['num_imgs']*config.settings['num_imgs'])
	
	newWindow = Toplevel(root)
	newWindow.title('Images found from {}'.format(config.settings['target_dataset']))
	photo = ImageTk.PhotoImage(Image.fromarray(pred_imgs))
	img_lbl = Label(newWindow,image=photo)
	img_lbl.image = photo
	img_lbl.pack()


def b1up(event):
	global b1, xold, yold
	global trainer_G
	global right_frame

	b1 = "up"
	xold = None        
	yold = None
	
	
	
	


def draw_img(event):
	global draw_image
	global b1
	if b1 == "down":
		global xold, yold
		global drawing_area
		
		if xold is not None and yold is not None:
			draw_image.line([(xold,yold),(event.x,event.y)],width = 3,fill='black')
			event.widget.create_line(xold,yold,event.x,event.y,width = 3, smooth=True,fill='black')
			
		xold = event.x
		yold = event.y


def undraw_img(event):
	global draw_image
	global drawing_area
	global img
	drawing_area.delete("all")
	img 		 = Image.new("RGB", (600, 600), 'white')
	draw_image =  ImageDraw.Draw(img)



def main():

	trainval()
	global clear_btn
	global search_btn
	global drawing_area
	drawing_area.grid(row=0,column=0,columnspan=2)
	clear_btn.grid(row=1,column=0)
	search_btn.grid(row=1,column=1)

	drawing_area.bind("<Motion>", draw_img)
	drawing_area.bind("<ButtonPress-1>", b1down)
	drawing_area.bind("<ButtonRelease-1>", b1up)
	clear_btn.bind("<ButtonPress-1>",undraw_img)
	search_btn.bind("<ButtonPress-1>",search_img)
	root.mainloop()



def trainval():
	global trainer_G
	print('initializing please wait!!!')
	trainer_G.load_weights_model()
	st0 = np.random.get_state()[1][0]
	trainer_G.set_mode(config.settings['mode']['val'])
	trainer_G.prepare_indices_domain_net()
	trainer_G.initialize_target_dataset(trainer_G.settings['target_dataset'])
	trainer_G.get_target_logits()
	
if __name__ == '__main__':
	main()



