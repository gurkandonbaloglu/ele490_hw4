import matplotlib.pyplot as plt
import numpy as np
import os

N, M = 512, 512

theta = np.pi / 4

output_path = 'q2_pi_over4'
if not os.path.exists(output_path):
    os.makedirs(output_path)
# Function to save images to the specified path
def save_images(image, title, filename, output_path):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')

    image_path = os.path.join(output_path, f'{filename}.png')
    plt.savefig(image_path)
    plt.close()
# Create a grid for generating the image
n_grid, m_grid = np.meshgrid(np.arange(N), np.arange(M))
f_nm = np.zeros((N,M))
# Create the edge in the image (upper part white, lower part black)
f_nm[m_grid < n_grid * np.tan(theta)] = 1
# Compute the 2D FFT of the original image and take the magnitude
f_D = np.abs(np.fft.fft2(f_nm))
f_C = np.abs(np.fft.fft2((-1)**(n_grid + m_grid) * f_nm))
f_L = np.log10(f_C + 1e-10)

f_M = f_nm - np.mean(f_nm)
f_DM = np.abs(np.fft.fft2(f_M))
f_CM = np.abs(np.fft.fft2((-1)**(n_grid + m_grid) * f_M))
f_LM = np.log10(f_CM + 1e-10)

images = [f_nm, f_D, f_C, f_L, f_M, f_DM, f_CM, f_LM]
titles = ['Spatial Domain f[n,m]', 'FFT |F_D|', 'Centered FFT |F_C|', 'Log Spectrum |F_L|',
          'Mean-Subtracted fM[n,m]', 'FFT |F_DM|', 'Centered FFT |F_CM|', 'Log Spectrum |F_LM|']

for i, (img, title) in enumerate(zip(images, titles)):
    filename = title.replace(' ', '_').replace('[', '').replace(']', '').replace('|', '').lower()
    save_images(img, title, filename, output_path)

print(f"Images have been saved in '{output_path}' directory.")


