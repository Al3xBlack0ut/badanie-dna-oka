import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pydicom.dataset import FileDataset

obraz='dno_oka/dataset/images/'
obraz+='Shepp_logan'
obraz+='.jpg'

def dno_oka():
    '''Główna funkcja'''
    with open(obraz, 'rb') as f:
        macierz = np.array(Image.open(f).convert('L'))
    plt.imshow(macierz, cmap='gray')
    plt.title('Dno oka')
    plt.axis('off')
    plt.show()