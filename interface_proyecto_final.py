from imutils.video import WebcamVideoStream
from tkinter import *
from tkinter import font
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
		self.pad_x = 5
		self.pad_y = 5
		self.color ='black'
		self.color1 ='green'
		self.color2 ='white'
		self.path = 'base'
		self.etiquetas = ['Desconocido', 'Valentina', 'Felipe', 'Kevyn', 'Jeferson', 'Alejandro']
		self.label_font = font.Font(family="Arial", size=15, weight='bold')
		self.framePrincipal = None
		self.frameSup = None
		self.frameVideo = None
		self.frameCapture = None
		self.frameInf = None
		self.lbl = None
		self.btn = None
		self.panelVideo = None
		self.panelCaptura = None
		self.thread = None
		self.stopEvent = None
		self.frame = None
		self.modelo = None
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

		self.frameVideo = Frame(self.frameSup)
		self.frameVideo.config(bg=self.color, width=self.d_width/2, height=self.d_height)
		self.frameVideo.pack(side='left', anchor='n')

		self.frameCapture = Frame(self.frameSup)
		self.frameCapture.config(bg=self.color, width=self.d_width/2, height=self.d_height)
		self.frameCapture.pack(side='right', anchor='n')


		#Frame Inferior
		self.frameInf = Frame(self.framePrincipal)
		self.frameInf.config(bg=self.color, width=self.d_width, height=self.d_height/4)
		self.frameInf.pack(anchor=CENTER)
		self.lbl = Label(self.frameInf, text='', fg=self.color2, bg=self.color, font= self.label_font, width=30)
		self.lbl.grid(row=0, column=0, columnspan=2, pady=self.pad_y)
		self.btnCapture = Button(self.frameInf, text = 'Capture', state=DISABLED, command=self.capturar_imagenes, activebackground='black', activeforeground='white', padx=10, pady=5)
		self.btnCapture.grid(row=1, column=0, pady=self.pad_y)
		self.btnStop = Button(self.frameInf, text = 'Stop', state=DISABLED, command= self.stop, activebackground='black', activeforeground='white', padx=10, pady=5)
		self.btnStop.grid(row=1, column=1, pady=self.pad_y)


	def cargar_modelo(self):
		if self.modelo == None:
			self.modelo = load_model('modelo.h5')

	def iniciar_video(self):
		self.vs.start()
		time.sleep(1)
		self.cargar_modelo()
		while not self.stopEvent.is_set():
			self.leer_imagen()
			_, _, img_color = self.crop_img(self.frame, True)
			self.asignar_panelVideo(img_color)
			self.btnCapture.configure(state='normal')
			self.btnStop.configure(state='normal')

	def leer_imagen(self):
		try:
			self.frame = self.vs.read()
		except Exception as e:
			print('[INFO] Error en la captura del video: ', e)

	def asignar_nombre(self, label, color):
		self.lbl.config(fg =color, text=label)

	def capturar_imagenes(self):
		boxes, frame_bboxes, img_color = self.crop_img(self.frame, False)
		if boxes==1:
			resultado_prediccion = self.prediccion(frame_bboxes)
			maximo = np.amax(resultado_prediccion)
			indice_etiqueta_prediccion = resultado_prediccion.tolist()[0].index(maximo)
			if self.etiquetas[indice_etiqueta_prediccion].upper() == 'DESCONOCIDO':
				self.asignar_nombre('Rostro NO reconocido.', 'yellow')
			else:
				self.asignar_nombre('Hola ' + self.etiquetas[indice_etiqueta_prediccion].upper() + ' bienvenid@.', 'green')
				date_now = datetime.datetime.now()
				filename = '{0}_{1}.png'.format(self.etiquetas[indice_etiqueta_prediccion].upper(), date_now.strftime('%Y-%m-%d_%H_%M_%S'))
				path_img = os.path.join(os.getcwd(), filename)
				img = self.convertir_arreglo_imagen(img_color, False)
				img.save(path_img)
				print('[INFO] saved {}'.format(filename))
				self.asignar_panelCapura(img_color)
		else:
			self.asignar_nombre('Rostro NO detectado', 'red')

	def crop_img(self, img, bounding_box):
		classifier = CascadeClassifier('haarcascade_frontalface_default.xml')
		img_color = img.copy()
		frame_bboxes = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		bboxes = classifier.detectMultiScale(image=frame_bboxes)

		if len(bboxes)==1:
			x,y,w,h = bboxes[0]
			if bounding_box:
				cv2.rectangle(img_color,(x,y),(x+w,y+h),(0,255,0),3)
			else:
				frame_bboxes = frame_bboxes[y:y+h, x:x+w]
				img_color = img_color[y:y+h, x:x+w]
				print('***Rostros detectados: {0}'.format(len(bboxes)))
		return len(bboxes), frame_bboxes, img_color

	def prediccion(self, img):
		frame_prediccion = cv2.resize(img, dsize=(120, 120), interpolation=cv2.INTER_CUBIC)
		frame_prediccion = img_to_array(frame_prediccion)
		frame_prediccion = (frame_prediccion-frame_prediccion.min())/(frame_prediccion.max()-frame_prediccion.min())
		frame_prediccion = np.array([frame_prediccion])
		resultado_prediccion = self.modelo.predict(frame_prediccion)
		return resultado_prediccion

	def stop(self):
		self.stopEvent.set()
		self.vs.stop()
		print('[INFO] Terminada la trasmision...')

	def convertir_arreglo_imagen(self, img, photo):
		img = cv2.resize(img, dsize=(int(self.d_width/2), int(self.d_width/2)), interpolation=cv2.INTER_CUBIC)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = Image.fromarray(img)
		img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
		if photo: img = ImageTk.PhotoImage(img)
		return img

	def asignar_panelVideo(self, captura_imagen):
		img = self.convertir_arreglo_imagen(captura_imagen, True)
		if self.panelVideo is None:
			self.panelVideo = Label(self.frameVideo, image=img)
			self.panelVideo.image = img
			self.panelVideo.pack(padx=self.pad_x, pady=self.pad_y)
		else:
			self.panelVideo.configure(image=img)
			self.panelVideo.image = img

	def asignar_panelCapura(self, captura_imagen):
		img = self.convertir_arreglo_imagen(captura_imagen, True)
		if self.panelCaptura is None:
			self.panelCaptura = Label(self.frameCapture, image=img)
			self.panelCaptura.image = img
			self.panelCaptura.pack(padx=self.pad_x, pady=self.pad_y)
		else:
			self.panelCaptura.configure(image=img)
			self.panelCaptura.image = img


ventana = Tk()
print("[INFO] Iniciando Interfaz y captura de video...")
vs = VideoStream(resolution=(300,300))

Interface(ventana, vs, 600, 200)
ventana.mainloop()