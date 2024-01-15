from PIL import Image
import glob


folder = './G_lr/'
# read all images in the folder
images = []
filenames = sorted(glob.glob(folder + '*.png'))
for filename in filenames:
    images.append(Image.open(filename))



# make gif
total_time = 5000
num_frame = len(filenames)
images[0].save(folder + '0.gif',
               save_all=True,
               append_images=images[1:],
               duration=total_time//num_frame,
               loop=0)

