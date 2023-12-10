import pickle
from django.shortcuts import render
from  django.core.files.storage import FileSystemStorage
from django.views.generic import TemplateView
from matplotlib import pyplot as plt 


#===============================================================================================
import os
import numpy as np 
import pickle
import random 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import cv2
import os

# import datasets
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import io, transforms
from torch.utils.data import Dataset, DataLoader, random_split

from transformers import Seq2SeqTrainer ,Seq2SeqTrainingArguments
from transformers import VisionEncoderDecoderModel , ViTFeatureExtractor
from transformers import AutoTokenizer ,  GPT2Config , default_data_collator




ENCODER = "google/vit-base-patch16-224"
DECODER = "gpt2"
feature_extractor = ViTFeatureExtractor.from_pretrained(ENCODER)
tokenizer = AutoTokenizer.from_pretrained(DECODER)
tokenizer.pad_token = tokenizer.unk_token
tokenizer = AutoTokenizer.from_pretrained('gpt2')
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
model = VisionEncoderDecoderModel.from_pretrained('./VIT_large_gpt2')



#===============================================================================================

def index(request ):
    if request.method =='POST':
        
        image = request.FILES['image']
        fs = FileSystemStorage()
        fs.save(image.name, image)
        test_image = os.path.join('media',image.name)
        img = Image.open(test_image).convert('RGB')

        try:
            
            inputs = feature_extractor(images= img, return_tensors="pt")

            outputs = model.generate(inputs['pixel_values'])
            caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return render(request, 'index.html',{'result':caption,"image":test_image})
           
        except Exception as err:
            fs.delete(test_image)
            return render(request, 'index.html',{'result':"bad image","image":test_image})

        

    return render(request, 'index.html')
# Create your views here.
