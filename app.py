import cv2

import numpy as np

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import Label, Button, Frame, Toplevel

from PIL import Image, ImageTk, ImageOps

from skimage.metrics import structural_similarity as ssim

import pywt

import os

class ImageDenoiseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Denoising App")
        self.root.geometry("800x600")

        self.frame = Frame(root)
        self.frame.pack()

        self.upload_button = Button(self.frame, text="Choose Image", command=self.choose_image)
        self.upload_button.pack(side=tk.LEFT)

        self.denoise_button = Button(self.frame, text="Denoise Image", command=self.denoise_image, state=tk.DISABLED)
        self.denoise_button.pack(side=tk.LEFT)

        self.download_button = Button(self.frame, text="Download Selected Image", command=self.download_image, state=tk.DISABLED)
        self.download_button.pack(side=tk.LEFT)

        self.original_image_label = Label(root)
        self.original_image_label.pack()

        self.denoised_images_frame = Frame(root)
        self.denoised_images_frame.pack()

        self.image_path = None
        self.denoised_images = []
        self.selected_image_path = None
        self.large_image_window = None

    def choose_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if self.image_path:
            self.display_image(self.image_path, self.original_image_label, (400, 300))
            self.denoise_button.config(state=tk.NORMAL)

    def display_image(self, image_path, label, size, border_color=None):
        image = Image.open(image_path)
        image.thumbnail(size, Image.LANCZOS)
        if border_color:
            image = ImageOps.expand(image, border=5, fill=border_color)
        photo = ImageTk.PhotoImage(image)
        label.config(image=photo)
        label.image = photo

    def denoise_image(self):
        if not self.image_path:
            messagebox.showerror("Error", "Please choose an image first!")
            return

        noisy_image = cv2.imread(self.image_path)
        noisy_image_rgb = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)

        # Applying different denoising techniques
        gaussian_denoised_img = cv2.GaussianBlur(noisy_image, (5, 5), 0)
        median_denoised_img = cv2.medianBlur(noisy_image, 5)
        bilateral_denoised_img = cv2.bilateralFilter(noisy_image, 9, 75, 75)
        nlm_denoised_img = cv2.fastNlMeansDenoisingColored(noisy_image, None, 10, 10, 7, 21)
        wavelet_denoised_img = self.wavelet_denoise(noisy_image_rgb)

        # Convert denoised images to RGB
        denoised_images = [
            cv2.cvtColor(gaussian_denoised_img, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(median_denoised_img, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(bilateral_denoised_img, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(nlm_denoised_img, cv2.COLOR_BGR2RGB),
            wavelet_denoised_img
        ]

        denoised_titles = ['Gaussian Blur', 'Median Blur', 'Bilateral Filter', 'Non-local Means', 'Wavelet Transform']

        # selecting the best denoised image using SSIM
        ssim_values = [ssim(noisy_image_rgb, img, win_size=7, channel_axis=2) for img in denoised_images]
        best_index = np.argmax(ssim_values)

        # Save images for display
        self.denoised_images = []
        for i, img in enumerate(denoised_images):
            img_path = f'denoised_{i}.png'
            cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            self.denoised_images.append((denoised_titles[i], img_path))

        self.best_denoised_image_path = self.denoised_images[best_index][1]
        self.selected_image_path = self.best_denoised_image_path

        self.display_denoised_images()
        self.download_button.config(state=tk.NORMAL)

    def wavelet_denoise(self, image_rgb):
        image_ycbcr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
        y, cb, cr = cv2.split(image_ycbcr)

        coeffs2 = pywt.dwt2(y, 'haar')
        LL, (LH, HL, HH) = coeffs2
        LL = pywt.threshold(LL, np.std(LL) / 2, mode='soft')
        LH = pywt.threshold(LH, np.std(LH) / 2, mode='soft')
        HL = pywt.threshold(HL, np.std(HL) / 2, mode='soft')
        HH = pywt.threshold(HH, np.std(HH) / 2, mode='soft')
        coeffs2 = LL, (LH, HL, HH)
        y_denoised = pywt.idwt2(coeffs2, 'haar')

        y_denoised = cv2.resize(y_denoised, (y.shape[1], y.shape[0]))

        image_ycbcr_denoised = cv2.merge((y_denoised.astype(np.uint8), cb, cr))
        image_rgb_denoised = cv2.cvtColor(image_ycbcr_denoised, cv2.COLOR_YCrCb2RGB)
        return image_rgb_denoised

    def display_denoised_images(self):
        for widget in self.denoised_images_frame.winfo_children():
            widget.destroy()

        for i, (title, img_path) in enumerate(self.denoised_images):
            label = Label(self.denoised_images_frame, text=title)
            label.grid(row=0, column=i)
            image_label = Label(self.denoised_images_frame)
            image_label.grid(row=1, column=i)
            image_label.bind("<Button-1>", lambda e, path=img_path: self.select_image(path, image_label))
            border_color = "red" if img_path == self.selected_image_path else None
            self.display_image(img_path, image_label, (200, 150), border_color)

    def select_image(self, img_path, label):
        self.selected_image_path = img_path
        self.display_denoised_images()
        self.show_large_image()

    def show_large_image(self):
        if self.large_image_window:
            self.large_image_window.destroy()

        self.large_image_window = Toplevel(self.root)
        self.large_image_window.title("Selected Image")

        original_label = Label(self.large_image_window, text="Original Image")
        original_label.pack()
        original_image_label = Label(self.large_image_window)
        original_image_label.pack()

        denoised_label = Label(self.large_image_window, text="Denoised Image")
        denoised_label.pack()
        denoised_image_label = Label(self.large_image_window)
        denoised_image_label.pack()

        self.display_image(self.image_path, original_image_label, (400, 300))
        self.display_image(self.selected_image_path, denoised_image_label, (400, 300))

    def download_image(self):
        if self.selected_image_path:
            save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
            if save_path:
                os.rename(self.selected_image_path, save_path)
                messagebox.showinfo("Success", "Image saved successfully!")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageDenoiseApp(root)
    root.mainloop()
