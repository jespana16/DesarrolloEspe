import unittest

import mock
from mock import patch
import numpy as np
import tkinter as tk
import threading

from interface_proyecto_final import Interface

class VideoStreamMock:
	def __init__(self):
		pass

	def start(self):
		pass

	def update(self):
		pass

	def read(self):
		pass

	def stop(self):
		pass


@patch('interface_proyecto_final.VideoStream', VideoStreamMock)
class PruebasFunciones(unittest.TestCase):
    def test_cargue_modelo(self):
        ventana = tk.Tk()
        vs = VideoStreamMock()
        inter = Interface(ventana, vs)
        self.assertTrue(inter.modelo is None)
        inter.cargar_modelo()
        self.assertTrue(inter.modelo is not None)

    def test_cargue_interface(self):
        ventana = tk.Tk()
        vs = VideoStreamMock()
        inter = Interface(ventana, vs)
        inter.start()
        inter.leer_imagen()
        inter.stop()
        self.assertIsInstance(inter.thread, threading.Thread)
        self.assertIsInstance(inter.lbl, tk.Label)
    
    def test_prediccion(self):
        ventana = tk.Tk()
        vs = VideoStreamMock()
        inter = Interface(ventana, vs)
        img = np.zeros([219, 219])
        inter.cargar_modelo()
        result = inter.prediccion(img)
        self.assertTrue(result.shape == (1,6))

'''
    def test_crop_img(self):
        #, classifier_mock
        ventana = tk.Tk()
        vs = VideoStreamMock()
        inter = Interface(ventana, vs)
        img = np.zeros((480, 640, 3), dtype=images.dtype)
        bounding_box = False
        len_bboxes, frame_bboxes, img_color = inter.crop_img(img, bounding_box)

        self.assertIsInstance(len_bboxes, int)
        self.assertIsInstance(frame_bboxes, np.array)
        self.assertIsInstance(img_color, np.array)
'''


if __name__ == '__main__':
    unittest.main()


