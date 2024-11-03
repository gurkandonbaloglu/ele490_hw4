import os
import numpy as np
import matplotlib.pyplot as plt

# Parameters
N, M = 512, 512
wr = np.pi / 32  # radial frequency, also give these wr values and change the output path np.pi / 32 and np.pi / 8

# Define angles and compute (ωy, ωx) pairs
angles = [0, np.pi / 6, np.pi / 3, np.pi / 2]
omega_pairs = [(wr * np.cos(theta), wr * np.sin(theta)) for theta in angles]

print("Computed (ωy, ωx) pairs:")
for i, (omega_x, omega_y) in enumerate(omega_pairs):
    print(f"θ = {angles[i]:.2f} radians -> (ωx, ωy) = ({omega_x:.4f}, {omega_y:.4f})")

# Create n, m meshgrid
n = np.arange(N)
m = np.arange(M)
n_grid, m_grid = np.meshgrid(n, m)

output_path = 'output_images_wr32' # change output path to save different adresses for other wr values

if not os.path.exists(output_path):
    os.makedirs(output_path)

def save_images(image, title, filename, output_path):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')

    image_path = os.path.join(output_path, f'{filename}.png')
    plt.savefig(image_path)
    plt.close()

# Function to generate the signal
def generate_cosine_signal(omega_x, omega_y):
    return np.cos(omega_x * n_grid + omega_y * m_grid) + 0.5

# Loop through each angle
for i, (omega_x, omega_y) in enumerate(omega_pairs):
    # Generate cosine signal
    f_nm = generate_cosine_signal(omega_x, omega_y)
    
    # Compute the 2D FFT
    f_D = np.abs(np.fft.fft2(f_nm))
    
    # Center the FFT
    f_c = np.abs(np.fft.fft2((-1)**(n_grid+m_grid)*f_nm))
    
    # Log magnitude spectrum for visualization
    f_l = np.log10(f_c + 1e-10)  # Adding a small constant to avoid log(0)
    
  
    save_images(f_nm, f'Spatial Domain: θ = {angles[i]:.2f} rad', f'spatial_theta_{i}', output_path)
    save_images(f_D, f'Original FFT |F_D|: θ = {angles[i]:.2f} rad', f'fft_original_theta_{i}', output_path)
    save_images(f_c, f'Centered FFT |F_C|: θ = {angles[i]:.2f} rad', f'fft_centered_theta_{i}', output_path)
    save_images(f_l, f'Log Spectrum |F_L|: θ = {angles[i]:.2f} rad', f'log_spectrum_theta_{i}', output_path)



