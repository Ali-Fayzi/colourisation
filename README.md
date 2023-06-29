# Unet-based Image Colorization
This project implements a neural network for image colorization using the popular Unet architecture. While Unet is typically used for image segmentation tasks, it can also be used for image reconstruction and colorization.

# Installation
To install the required packages, run:
```
pip install -r requirements.txt
```
# Usage
To use the trained network to generate colorized images, run:
```
python inference.py --weight_path ./weights/weight.pt --input_path ./images --output_path ./output --plot
```
You can specify the path to the trained weights, the input directory containing grayscale images, the output directory to save colorized images, and whether or not to display the images.

# Output
         Gray Scale Image		         Random Pixel		       Output	              Ground Truth	
![Out1](https://github.com/Ali-Fayzi/colourisation/blob/master/output/images_pexels-jovana-nesic-593655.jpg?raw=true)
![Out2](https://github.com/Ali-Fayzi/colourisation/blob/master/output/images_1162413.jpg?raw=true)
![Out3](https://github.com/Ali-Fayzi/colourisation/blob/master/output/images_1162415.jpg?raw=true)
![Out4](https://github.com/Ali-Fayzi/colourisation/blob/master/output/images_live_.cid.png?raw=true)
![Out5](https://github.com/Ali-Fayzi/colourisation/blob/master/output/images_pexels-suparada-intharoek-1767434.jpg?raw=true)

# Training
The network was trained on the COCO dataset using a customized dataloader that randomly selects only 0.1% of the actual image pixels for conditional colorization. Due to limited resources, the network was trained for only 2 epochs.

# Acknowledgements
This project was implemented by [Ali Fayzi](https://www.linkedin.com/in/ali-fayzi-28433a138/). Feel free to use and modify the code for your own purposes. If you have any questions or suggestions, please feel free to contact me.
