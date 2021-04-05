from imutils.video import WebcamVideoStream
from tkinter import *
from PIL import ImageTk, Image, ImageEnhance
import numpy as np
import datetime
import imutils
import time
import cv2
from cv2 import CascadeClassifier
import os
import threading
from keras.models import load_model
from keras.preprocessing.image import img_to_array

class VideoStream:
	def __init__(self, src=0, usePiCamera=False, resolution=(300, 300), framerate=32):
		if usePiCamera:
			from pivideostream import PiVideoStream
			self.stream = PiVideoStream(resolution=resolution,framerate=framerate)
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
	def __init__(self, ventana, vs, w=600, h=300):
		self.vs = vs
		self.root = ventana
		self.d_width = w
		self.d_height = h
		self.i = 1
		self.color ='black'
		self.color1 ='green'
		self.color2 ='white'
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
		self.img2 = None
		self.thread = None
		self.stopEvent = None
		self.filename = None
		self.frame = None
		self.frame_prediccion = None
		self.modelo = None
		self.pad_x = 2
		self.pad_y = 2
		self.resultado_prediccion = None
		self.etiquetas = ['Desconocido', 'Valentina', 'Felipe', 'Kevyn', 'Jeferson', 'Alejandro']
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
		self.frameSup.config(bg=self.color, width=self.d_width, height=self.d_height)
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
		self.lbl = Label(self.frameInf, text='', fg=self.color2, bg=self.color)
		self.lbl.pack()
		self.btnCapture = Button(self.frameInf, text = 'Capture', state=DISABLED, command=self.capturar_imagenes, activebackground='black', activeforeground='white', padx=10, pady=5)
		self.btnCapture.pack(side='left')
		self.btnStop = Button(self.frameInf, text = 'Stop', state=DISABLED, command= self.stop, activebackground='black', activeforeground='white', padx=10, pady=5)
		self.btnStop.pack(side='left')

	def asignar_nombre(self, label, color):
		self.lbl.config(fg =color, text=label)

	def leer_imagen(self):
		try:
			self.frame = self.vs.read()
		except Exception as e:
			print('[INFO] Error en la captura del video: ', e)

	def capturar_imagenes(self):
		date_now = datetime.datetime.now()
		self.filename = '{}.png'.format(date_now.strftime('%Y-%m-%d_%H_%M_%S'))
		path_img = os.path.join(os.getcwd(), self.filename)
		
		#FaceDetection
		boxes = self.crop_img()
		if boxes==1:
			self.prediccion()
			self.convertir_arreglo_imagen()
			self.img.save(path_img)
			print('[INFO] saved {}'.format(self.filename))
			maximo = np.amax(self.resultado_prediccion)
			indice_etiqueta_prediccion = self.resultado_prediccion.tolist()[0].index(maximo)
			if self.etiquetas[indice_etiqueta_prediccion].upper() == 'DESCONOCIDO':
				self.asignar_nombre('No se ha identificado un rostro.', 'yellow')
			else:
				self.asignar_nombre('Hola ' + self.etiquetas[indice_etiqueta_prediccion] + ' bienvenido.', 'green')
			self.asignar_panelCapura()
		else:
			self.asignar_nombre('Rostro no detectado', 'red')
			self.frame_prediccion = None
			self.asignar_panelCapura()

	def prediccion(self):
		self.frame_prediccion = cv2.resize(self.frame_bboxes, dsize=(120, 120), interpolation=cv2.INTER_CUBIC)
		self.frame_prediccion = img_to_array(self.frame_prediccion)
		self.frame_prediccion = (self.frame_prediccion-self.frame_prediccion.min())/(self.frame_prediccion.max()-self.frame_prediccion.min())
		self.frame_prediccion = np.array([self.frame_prediccion])
		self.resultado_prediccion = self.modelo.predict(self.frame_prediccion)
		self.frame_prediccion = cv2.cvtColor(self.frame_prediccion, cv2.COLOR_GRAY2RGB)

	def crop_img(self):
		classifier = CascadeClassifier('haarcascade_frontalface_default.xml')
		self.frame_bboxes = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)
		bboxes = classifier.detectMultiScale(image=self.frame_bboxes)
		print('***Cantidad Boxes detectados: {0} - {1}'.format(len(bboxes), bboxes))

		if len(bboxes)==1:
			x,y,w,h = bboxes[0]
			self.frame_bboxes = self.frame_bboxes[y:y+h, x:x+w]
			#cv2.rectangle(self.frame_bboxes,(x,y),(x+w,y+h),(255,0,0),3)
		return len(bboxes)

	def cargar_modelo(self):
		if self.modelo == None:
			self.modelo = load_model('modelo.h5')

	def iniciar_video(self):
		self.vs.start()
		time.sleep(1)
		self.cargar_modelo()
		while not self.stopEvent.is_set():
			self.leer_imagen()
			self.asignar_panelVideo()
			self.btnCapture.configure(state='normal')
			self.btnStop.configure(state='normal')

	def stop(self):
		self.stopEvent.set()
		self.vs.stop()
		print('[INFO] Terminada la trasmision...')

	def convertir_arreglo_imagen(self, img):
		img = cv2.resize(img, dsize=(int(self.d_width/2), int(self.d_width/2)), interpolation=cv2.INTER_CUBIC)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = Image.fromarray(img)
		img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
		return img

	def asignar_panelVideo(self):
		_img = self.convertir_arreglo_imagen(self.frame)
		img = ImageTk.PhotoImage(_img)
		if self.panelVideo is None:
			self.panelVideo = Label(self.frame, image=img)
			self.panelVideo.image = img
			self.panelVideo.pack(padx=self.pad_x, pady=self.pad_y)
		else:
			self.panelVideo.configure(image=img)
			self.panelVideo.image = img

	def asignar_panelCapura(self):
		_img = self.convertir_arreglo_imagen(self.frame_prediccion)
		img = ImageTk.PhotoImage(_img)
		if self.panelCaptura is None:
			self.panelCaptura = Label(_img, image=img)
			self.panelCaptura.image = img
			self.panelCaptura.pack(padx=self.pad_x, pady=self.pad_y)
		else:
			self.panelCaptura.configure(image=img)
			self.panelCaptura.image = img


ventana = Tk()
print("[INFO] Iniciando Interfaz y captura de video...")
vs = VideoStream(resolution=(300,300))
time.sleep(2.0)

Interface(ventana, vs, 600, 200)
ventana.mainloop()