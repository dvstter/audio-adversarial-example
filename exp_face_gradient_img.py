from facenet_pytorch import InceptionResnetV1 as Resnet
from PIL import Image
from torchvision import transforms
import torch as T
import torch.nn.functional as F
import numpy as np


def convert_flattened_index(flt_idx, width):
  return [[idx//width, idx%width] for idx in flt_idx]

def get_modified_picture(file_path, most_or_least, modifications, save_path, show=True):
  # get gradients
  resnet = Resnet(pretrained='vggface2').eval()
  img = Image.open(file_path)
  trans_method = transforms.ToTensor()
  tensor = trans_method(img).unsqueeze(0)
  tensor = T.tensor(tensor, requires_grad=True)
  _, _, height, width = tensor.shape
  output = resnet(tensor)
  label = T.LongTensor([0])
  loss = F.cross_entropy(output, label)
  resnet.zero_grad()
  loss.backward()


  # process gradients and get indexj
  grads = abs(tensor.grad.data.numpy())
  grads[0, 0] += grads[0, 1] + grads[0, 2]
  grads = grads[0, 0]
  flattened_grads = grads.reshape(-1)
  flattened_index = np.argsort(flattened_grads)
  if most_or_least == 'most':
    flattened_index = np.flip(flattened_index)
  index = convert_flattened_index(flattened_index, width)

  # modify the picture
  img_array = np.array(img) # [width, height, channels]
  img_array = img_array.transpose([2,1,0]) # [channels, height, width]
  for idx in index[:modifications]:
    img_array[tuple([0]+idx)] = 255 if most_or_least == 'most' else 0
    img_array[tuple([1]+idx)] = 255 if most_or_least == 'least' else 0
    img_array[tuple([2]+idx)] = 0

  img_array = img_array.transpose([2,1,0]) # [width, height, channels]
  new_img = Image.fromarray(img_array)
  new_img.save(save_path)
  if show:
    new_img.show()

if __name__ == '__main__':
  get_modified_picture('/Users/yanghanlin/Downloads/CASIA-WebFace/0650751/010.jpg', 'most', 10000, '/Users/yanghanlin/Downloads/img1_most.jpg')
  get_modified_picture('/Users/yanghanlin/Downloads/CASIA-WebFace/0650751/010.jpg', 'least', 10000, '/Users/yanghanlin/Downloads/img1_least.jpg')
  get_modified_picture('/Users/yanghanlin/Downloads/CASIA-WebFace/0650751/009.jpg', 'most', 10000, '/Users/yanghanlin/Downloads/img2_most.jpg')
  get_modified_picture('/Users/yanghanlin/Downloads/CASIA-WebFace/0650751/009.jpg', 'least', 10000, '/Users/yanghanlin/Downloads/img2_least.jpg')
  get_modified_picture('/Users/yanghanlin/Downloads/CASIA-WebFace/0650751/008.jpg', 'most', 10000, '/Users/yanghanlin/Downloads/img3_most.jpg')
  get_modified_picture('/Users/yanghanlin/Downloads/CASIA-WebFace/0650751/008.jpg', 'least', 10000, '/Users/yanghanlin/Downloads/img3_least.jpg')

