# FRED-Lite Network for 5G-LTE Spectrogram-based Spectrum Sensing

Spectrum sensing is a key component in cognitive radio networks, responsible for detecting available spectrum bands and facilitating efficient, adaptive spectrum management in increasingly dense and dynamic wireless communication environments. However, existing segmentation models often prioritize accuracy at the expense of computational efficiency, resulting in complex architectures with extensive connections and heavy reliance on convolutional layers. Such complexity hinders their applicability in real-time scenarios and resource-constrained embedded systems. 

To overcome these limitations, we introduce FRED-Lite (Full-Resolution Encoder-Decoder Lite) network, a lightweight yet effective deep learning architecture specifically tailored for semantic segmentation of 5G and LTE signals in spectrograms generated via short-time Fourier transforms, capturing both temporal and spectral features with high fidelity.

FRED-Lite integrates a full-resolution encoder structure, a boundary refinement mechanism within the decoder, and a grouped multi-kernel input extractor module. This architectural design enables efficient spectral feature extraction by jointly learning local and global representations, thereby improving segmentation accuracy and ensuring robust performance across a wide range of spectrogram conditions and signal environments.

The Python code and dataset provided here are part of the accepted paper in the First International Conference On Computational Intelligence In Engineering Science (ICCIES), Ho Chi Minh City, Vietnam, July 2025.

Huu-Tai Nguyen, Gia-Phat Hoang, Hai-Trang Phuoc Dang, and Thien Huynh-The, "A Lightweight Full-Resolution Encoder-Decoder Network for 5G-LTE Spectrogram-based Spectrum Sensing," in Proc. ICCIES, Jul. 2025.

The dataset can be downloaded from [Kaggle]([https://www.kaggle.com/datasets/huutai23012003/spectrum-sesing-dataset/data](https://www.kaggle.com/datasets/huutai23012003/c02-dataset/settings)). Please report if it is not available.

If there are any errors or topics that need to be discussed, please contact [Huu-Tai Nguyen](https://github.com/HuuTaiNg) via email at n.huutai231@gmail.com.
