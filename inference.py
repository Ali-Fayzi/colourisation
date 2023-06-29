import os 
import cv2
import torch
import argparse
import numpy as np
from network import ColourisationNetwork
from glob import glob 
from matplotlib import pyplot as plt 
class Colour:
  def __init__(self,device ,weight_path):
    self.device = device
    self.network = ColourisationNetwork(in_channels=4, out_channels=3)
    state_dict = torch.load(weight_path, map_location=device)
    self.network.load_state_dict(state_dict)
    self.network = self.network.eval()
    self.network = self.network.to(device)
  def image_resize(self,img, size, interpolation):
    h, w = img.shape[:2]
    c = None if len(img.shape) < 3 else img.shape[2]
    if h == w: return cv2.resize(img, (size, size), interpolation),0,0,w,h
    if h > w: dif = h
    else:     dif = w
    x_pos = int((dif - w)/2.)
    y_pos = int((dif - h)/2.)
    if c is None:
      mask = np.zeros((dif, dif), dtype=img.dtype)
      mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
      mask = np.zeros((dif, dif, c), dtype=img.dtype)
      mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
    return cv2.resize(mask, (size, size), interpolation),x_pos,y_pos,w,h
  def preprocess(self,image):
    img_h  , img_w = image.shape[0] , image.shape[1]
    image ,x_pos,y_pos,w,h = self.image_resize(image, 640, interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32)
    image /= 255.0
    if len(image.shape) <=2:
      image = np.expand_dims(image,2)
    image = image.transpose(2, 0, 1)
    image_tensor = torch.from_numpy(image.astype(np.float32))
    image_tensor = torch.unsqueeze(image_tensor, dim=0)
    image_tensor = image_tensor.to(self.device)
    return image_tensor ,x_pos,y_pos,w,h,img_h,img_w
  def postprocess(self,y_pred,x_pos,y_pos,w,h,img_h,img_w):
    if img_h>=img_w:
      size= img_h
    else:
      size=img_w
    y_pred_np = y_pred.detach().cpu().numpy()
    y_pred_np = y_pred_np.transpose(2,3,1,0)
    y_pred_np = np.reshape(y_pred_np,(640,640,3))
    y_pred_np , _ , _ , _ , _  = self.image_resize(y_pred_np, size, interpolation=cv2.INTER_AREA)
    y_pred_np = y_pred_np[y_pos:y_pos+h , x_pos:x_pos+w]
    return y_pred_np 
  def run(self,gray_image , random_pixel):
    with torch.set_grad_enabled(False):
      gray_image,x_pos,y_pos,w,h,img_h,img_w  = self.preprocess(gray_image)
      random_pixel,x_pos,y_pos,w,h,img_h,img_w  = self.preprocess(random_pixel)
      x = torch.cat((gray_image, random_pixel), dim=1)
      y_pred = self.network(x)
      image  = self.postprocess(y_pred,x_pos,y_pos,w,h,img_h,img_w)
      return image       
# Test Code
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Colourisation Application')
  parser.add_argument('--weight_path',default='./weights/weight.pt', type=str,
                      help='select network weight')
  parser.add_argument('--input_path',default='./images', type=str,
                      help='select test images folder')
  parser.add_argument('--output_path',default='./output', type=str,
                      help='select output folder')
  parser.add_argument('--plot',default=False, action=argparse.BooleanOptionalAction, 
                      help='show output')
  args = parser.parse_args()
  print(args)
  if not os.path.exists(args.input_path):
    print(f"Selected Folder Not Exists!")
  if not os.path.exists(args.output_path) and args.output_path!='':
    os.makedirs(args.output_path)
    print(f"Create Output Folder | Path:{args.output_path}")
  device = torch.device("cpu" if not torch.cuda.is_available() else 'cuda')
  weight = args.weight_path
  colour = Colour(device,weight)
  images = glob(os.path.join(args.input_path,"*"))
  for index,image_name in enumerate(images):
    image = cv2.imread(image_name)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    random_matrix = np.random.rand(*image.shape[:2]) * 100
    random_pixel = image.copy()
    random_pixel[random_matrix >= 0.1] = 0
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray_image = np.expand_dims(gray_image,2)
    segmentImage = colour.run(gray_image=gray_image,random_pixel=random_pixel)
    if args.plot:
      plt.imshow(np.hstack([cv2.cvtColor(gray_image,cv2.COLOR_GRAY2BGR)/255,random_pixel/255,segmentImage,image/255]))
      plt.xlabel(f"(GrayImage) , (RandomPixel) , (Output) , (GroundTruth)")
      plt.show()
    if args.output_path !='':
      image_out = image_name.split('/')[-1].replace('\\','_')
      plt.imsave(f"{args.output_path}/{image_out}",np.hstack([cv2.cvtColor(gray_image,cv2.COLOR_GRAY2BGR)/255,random_pixel/255,segmentImage,image/255]))
    print(f"Index:{index+1}/{len(images)} Done!")