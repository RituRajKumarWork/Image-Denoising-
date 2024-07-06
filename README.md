# Image Denoising App

This project is a simple GUI-based application for image denoising using various techniques. The application is built using Python and Tkinter for the GUI, with OpenCV, NumPy, and other libraries for image processing.

## Features

- Load an image from your computer.
- Apply different denoising techniques:
  - Gaussian Blur
  - Median Blur
  - Bilateral Filter
  - Non-local Means Denoising
  - Wavelet Transform-based Denoising
- Display and compare original and denoised images.
- Save the denoised image to your computer.

## Installation

To run this project, you need to install the required dependencies. Use the `requirements.txt` file to install them.

### Requirements

- Python 3.x
- OpenCV
- NumPy
- Tkinter
- Pillow
- Scikit-Image
- PyWavelets

### Installation Steps

1. Clone the repository:
    ```sh
    git clone https://github.com/RituRajKumarWork/Image-Denoising-.git
    cd Image-Denoising-
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Run the application:
    ```sh
    python app.py
    ```

## Usage

- Click on "Choose Image" to load an image.
- Click on "Denoise Image" to apply denoising techniques.
- Click on any of the denoised images to view a larger comparison with the original image.
- Click on "Download Selected Image" to save the selected denoised image.

## License

This project is licensed under the MIT License.

## Contributing

Feel free to fork this repository and contribute by submitting pull requests. Any contributions, issues, and feature requests are welcome!

## Contact

For any questions, feel free to contact me through my [GitHub profile](https://github.com/RituRajKumarWork).
