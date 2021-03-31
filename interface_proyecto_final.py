#Imports
from imutils.video import WebcamVideoStream
from tkinter import *
from PIL import ImageTk, Image, ImageEnhance
import numpy as np
import datetime
import imutils
import time
import cv2
import os
import threading

class VideoStream:
	def __init__(self, src=0, usePiCamera=False, resolution=(300, 300), framerate=32):
		if usePiCamera:
			from pivideostream import PiVideoStream
			self.stream = PiVideoStream(resolution=resolution,
				framerate=framerate)
		else:
			self.stream = WebcamVideoStream(src=src)

	def start(self):
		return self.stream.start()

	def update(self):
		self.stream.update()

	def read(self):
		return self.stream.read()

	def stop(self):
		self.stream.stop()


class Interface:
	def __init__(self, ventana, vs):
		self.vs = vs
		self.root = ventana
		self.d_width=600
		self.d_height=300
		self.i=1
		self.color='black'
		self.color1='green'
		self.color2='white'
		self.path = 'base'
		self.framePrincipal = None
		self.frameSup = None
		self.framevideo = None
		self.framecapture = None
		self.frameInf = None
		self.lbl = None
		self.btn = None
		self.panelVideo = None
		self.panelCaptura = None
		self.img = None
		self.thread = None
		self.stopEvent = None
		self.pad_x=2
		self.pad_y=2
		self.position()
		self.start()
		self.leer_imagen()

	def position(self):		 
		positionRight = int(self.root.winfo_screenwidth()/2 - self.d_width/2)
		positionDown = int(self.root.winfo_screenheight()/2 - self.d_height/2)
		self.root.geometry('+{}+{}'.format(positionRight, positionDown))

	def start(self):
		self.root.title('Face_Recognition')
		self.root.resizable(False, False)
		self.root.iconbitmap('icono.ico')
		self.root.config(bg='black')

		self.stopEvent = threading.Event()
		self.thread = threading.Thread(target=self.iniciar_video, args=())
		self.thread.start()


		#Frame Principal
		self.framePrincipal = Frame(self.root)
		self.framePrincipal.config(bg=self.color, width=self.d_width, height=self.d_height)
		self.framePrincipal.pack(fill='both', expand='False')


		#Frame Superior
		self.frameSup = Frame(self.framePrincipal)
		self.frameSup.config(width=self.d_width, height=self.d_height)
		self.frameSup.pack(fill='both', expand='False')

		self.framevideo = Frame(self.frameSup)
		self.framevideo.config(bg=self.color, width=self.d_width/2, height=self.d_height)
		self.framevideo.pack(side='left', anchor='n')

		self.framecapture = Frame(self.frameSup)
		self.framecapture.config(bg=self.color, width=self.d_width/2, height=self.d_height)
		self.framecapture.pack(side='right', anchor='n')


		#Frame Inferior
		self.frameInf = Frame(self.framePrincipal)
		self.frameInf.config(bg=self.color, width=self.d_width, height=self.d_height/4)
		self.frameInf.pack()
		self.lbl = Label(self.frameInf, text='NombrePersona', fg=self.color2, bg=self.color)
		self.lbl.pack()
		self.btnVideo = Button(self.frameInf, text = 'Video', command=self.iniciar_video, activebackground='black', activeforeground='white', padx=10, pady=5)
		self.btnVideo.pack(side='left')
		self.btnCapture = Button(self.frameInf, text = 'Capture', state=DISABLED, command=self.capturar_imagenes, activebackground='black', activeforeground='white', padx=10, pady=5)
		self.btnCapture.pack(side='left')
		self.btnStop = Button(self.frameInf, text = 'Stop', state=DISABLED, command= self.stop, activebackground='black', activeforeground='white', padx=10, pady=5)
		self.btnStop.pack(side='left')

	def asignar_nombre(self):
		global i
		self.lbl.config(text='NombrePersona' + str(self.i))
		self.i+=1

	def leer_imagen(self):
		try:
			frame = self.vs.read()
			frame = imutils.resize(frame, width=int(self.d_width/2-self.pad_x), height=int(self.d_height-self.pad_y))
			self.img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			self.img = Image.fromarray(self.img)
		except:
			print('[INFO] Error en la captura del video...')

	def capturar_imagenes(self):
		date_now = datetime.datetime.now()
		filename = '{}.png'.format(date_now.strftime('%Y-%m-%d_%H_%M_%S'))
		path_img = os.path.join(os.getcwd(), filename)
		self.img.save(path_img)
		print('[INFO] saved {}'.format(filename))
		self.asignar_nombre()
		self.asignar_panelCapura()

	def iniciar_video(self):
		while not self.stopEvent.is_set():
			self.vs.start()
			self.leer_imagen()
			self.asignar_panelVideo()
			self.btnCapture.configure(state='normal')
			self.btnStop.configure(state='normal')

	def stop(self):
		self.stopEvent.set()
		self.vs.stop()
		print('[INFO] Terminada la trasmision...')

	def asignar_panelVideo(self):
		img = ImageTk.PhotoImage(self.img)
		if self.panelVideo is None:
			self.panelVideo = Label(self.framevideo, image=img)
			self.panelVideo.image = img
			self.panelVideo.pack(padx=self.pad_x, pady=self.pad_y)
		else:
			self.panelVideo.configure(image=img)
			self.panelVideo.image = img

	def asignar_panelCapura(self):
		img = ImageTk.PhotoImage(self.img)
		if self.panelCaptura is None:
			self.panelCaptura = Label(self.framecapture, image=img)
			self.panelCaptura.image = img
			self.panelCaptura.pack(padx=self.pad_x, pady=self.pad_y)
		else:
			self.panelCaptura.configure(image=img)
			self.panelCaptura.image = img


ventana = Tk()
print("[INFO] Iniciando Interfaz y captura de video...")
vs = VideoStream()
time.sleep(2.0)

Interface(ventana, vs)
ventana.mainloop()