# Sign Language Recognition System

![Sign Language Recognition Demo](assets/demo.gif)

A sophisticated system for real-time sign language recognition, utilizing Convolutional Neural Networks (CNNs) and computer vision to translate hand gestures into text or speech. This project aims to enhance accessibility by enabling seamless communication for the deaf and hard-of-hearing community.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
  - [Creating Gestures](#creating-gestures)
  - [Displaying Gestures](#displaying-gestures)
  - [Training the Model](#training-the-model)
  - [Generating Model Reports](#generating-model-reports)
  - [Testing Gestures](#testing-gestures)
  - [Interactive Modes](#interactive-modes)
- [Dataset](#dataset)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview
The Sign Language Recognition System employs advanced deep learning techniques to recognize and interpret sign language gestures in real time. Built with a focus on accuracy and usability, it supports the American Sign Language (ASL) alphabet, numbers, and custom gestures. The system integrates computer vision for gesture detection and a CNN-based model for classification, making it a powerful tool for accessibility and communication.

This project combines elements of gesture capture, model training, and real-time recognition, with additional features like text-to-speech conversion and a calculator mode for interactive applications.

## Features
- **Real-Time Recognition**: Translates sign language gestures into text or speech using a webcam.
- **Comprehensive Gesture Set**: Supports 44 gestures, including 26 ASL letters, 10 numbers, and additional custom gestures.
- **Interactive Modes**: Includes text mode for forming words and a calculator mode for performing arithmetic and bitwise operations.
- **High Accuracy**: Utilizes a CNN model trained on a large dataset of grayscale images.
- **Customizable Gestures**: Allows users to add or replace gestures with a streamlined capture process.
- **Text-to-Speech**: Converts recognized gestures into spoken words for enhanced accessibility.
- **Cross-Platform**: Compatible with Windows, macOS, and Linux.

## Technologies Used
- **Python 3.x**: Core programming language.
- **TensorFlow 1.5 / Keras**: For building and training the CNN model.
- **OpenCV 3.4**: For real-time image processing and gesture capture.
- **h5py**: For handling model weights and data storage.
- **pyttsx3**: For text-to-speech functionality.
- **NumPy**: For numerical computations.
- **Hardware**: Optimized for CPUs, with optional GPU support for faster training.

## Installation
To set up the project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/sign-language-recognition.git
   cd sign-language-recognition
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   - For GPU users (requires NVIDIA GPU and TensorFlow GPU prerequisites):
     ```bash
     pip install -r requirements_gpu.txt
     ```
   - For CPU users:
     ```bash
     pip install -r requirements_cpu.txt
     ```

4. **Verify Installation**:
   Ensure dependencies are installed correctly:
   ```bash
   python -m scripts.verify_installation
   ```

## Usage
### Creating Gestures
1. **Set Hand Histogram**:
   Calibrate the system for your skin tone and lighting conditions:
   ```bash
   python set_hand_hist.py
   ```
   - A "Set hand histogram" window with a 5x10 grid will appear.
   - Place your hand to cover all squares and press `c` to generate a threshold image.
   - Verify that the "Thresh" window shows white patches corresponding to your skin tone.
   - Press `s` to save the histogram once satisfied.

2. **Capture New Gestures** (Optional):
   To add or replace gestures:
   ```bash
   python create_gestures.py
   ```
   - Enter the gesture number and name when prompted.
   - Perform the gesture within the green box in the "Capturing gestures" window.
   - Press `c` to start capturing (1200 images per gesture). Pause/resume with `c`.
   - After capturing, flip images to augment the dataset:
     ```bash
     python flip_images.py
     ```
   - Process the images for training:
     ```bash
     python load_images.py
     ```

### Displaying Gestures
View all stored gestures in the `gestures/` folder:
```bash
python display_all_gestures.py
```

### Training the Model
Train the CNN model using either TensorFlow or Keras:
- **TensorFlow**:
  ```bash
  python cnn_tf.py
  ```
  Model checkpoints and metagraph are saved in `tmp/cnn_model3/`.
- **Keras**:
  ```bash
  python cnn_keras.py
  ```
  The model is saved as `cnn_model_keras2.h5` in the root directory.

Retrain only when adding or removing gestures.

### Generating Model Reports
Evaluate model performance:
1. Ensure `test_images` and `test_labels` are generated by `load_images.py`.
2. Run:
   ```bash
   python get_model_reports.py
   ```
   Outputs include confusion matrix, F-scores, precision, and recall.

### Testing Gestures
1. Set the hand histogram (if not already done):
   ```bash
   python set_hand_hist.py
   ```
2. Start gesture recognition:
   ```bash
   python recognize_gesture.py
   ```
   Perform gestures within the green box for real-time recognition.

### Interactive Modes
Run the interactive application:
```bash
python fun_util.py
```
- **Text Mode** (`t`):
  - Form words using finger spellings or predefined gestures.
  - Maintain each gesture for 15 frames to register.
  - Text is converted to speech when the hand is removed from the green box.
- **Calculator Mode** (`c`):
  - Confirm digits by holding gestures for 20 frames.
  - Confirm numbers with the "best of luck" gesture for 25 frames.
  - Select operators (e.g., 1 for `+`, 2 for `-`, up to 10 operators including bitwise operations).

## Dataset
The dataset includes 44 gestures (26 ASL letters, 10 numbers, and additional custom gestures), with 2400 grayscale images (50x50 pixels) per gesture, created by capturing 1200 images and flipping them. Images are stored in the `gestures/` folder. To add new gestures, follow the [Creating Gestures](#creating-gestures) section.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request with a clear description.

Adhere to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Author
- **Shuban Borkar**
- GitHub: [shubanborkar](https://github.com/shubanborkar)
- Email: [shubanborkar@gmail.com](mailto:shubanborkar@gmail.com)
---

Thank you for exploring the Sign Language Recognition System! Let's work together to make communication more inclusive.