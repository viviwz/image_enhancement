from PIL import Image
from ipywidgets import interact, fixed
import matplotlib.pyplot as plt

def display_image(x):
    # x_scaled = np.uint8(255 * (x - x.min()) / x.ptp())
    x_scaled = x
    # plt.imshow(x, cmap='gray')
    return Image.fromarray(x_scaled)
    return 
    
def display_sequence(images):
    def _show(frame=(0, len(images)-1)):
        return display_image(images[frame,...])
    return interact(_show)