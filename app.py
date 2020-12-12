import streamlit as st

from PIL import Image
import os, shutil
import matplotlib.pyplot as plt
import numpy

from width_control import *

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable

st.set_page_config(
  page_title="FoodRescue",
  page_icon=":seedling:",
  layout="centered",
  initial_sidebar_state="collapsed",
)
select_block_container_style()

# -------------------- Header --------------------

st.markdown('# FoodRescue - Rescuing your Food.')



# -------------------- Helper Functions --------------------

data_dir = 'data/dataset/test'

test_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                    ])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('fruitmodel.pth')
model.eval()

classes = datasets.ImageFolder(data_dir, transform=test_transforms).classes

@st.cache
def predict_image(image):
  image_tensor = test_transforms(image).float()
  image_tensor = image_tensor.unsqueeze_(0)
  input = Variable(image_tensor)
  input = input.to(device)
  output = model(input)
  index = output.data.cpu().numpy().argmax()
  return index

@st.cache
def get_random_images(num):
  data = datasets.ImageFolder(data_dir, transform=test_transforms)
  classes = data.classes
  indices = list(range(len(data)))
  np.random.shuffle(indices)
  idx = indices[:num]
  from torch.utils.data.sampler import SubsetRandomSampler
  sampler = SubsetRandomSampler(idx)
  loader = torch.utils.data.DataLoader(data, 
              sampler=sampler, batch_size=num)
  dataiter = iter(loader)
  images, labels = dataiter.next()
  return images, labels


# -------------------- Upload --------------------

st.markdown('<hr>', unsafe_allow_html=True)

st.markdown('## Is your apple/orange/banana spoilt?')
st.markdown('## 🍎🍊🍌')

img = None
uploaded_file = st.file_uploader('')

if uploaded_file is not None:
  img = Image.open(uploaded_file)
  # model only support 3 color channel image
  if img.mode in ("RGBA", "P"): img = img.convert("RGB")
  st.success('Uploaded!')

# -------------------- Display --------------------

img_display = st.empty()
col_width = False

if img:
  pred = classes[predict_image(img)]
  img_display.image(img, use_column_width=col_width)
  st.markdown('# {}'.format(pred))




