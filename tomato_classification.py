#%%
from transformers import pipeline
classifier = pipeline(model="openai/clip-vit-large-patch14")

#%%
import os
main_dir = "images"
labels = []
ix = 0 
for image in os.listdir(main_dir):
    image_path = os.path.join(main_dir,image)
    result = classifier(image_path,candidate_labels=["red riped tomato","orange semi riped tomato","green unriped tomato"])[0]
    labels.append(result['label'])
    print(f"done {ix}")
    ix+=1
# %%
riped = 0
semi = 0
unriped = 0
for label in labels:
    if label == 'red ripe tomato':
        riped+=1
    elif label == 'green unriped tomato':
        unriped+=1
    else: 
        semi+=1
#%%
total = len(os.listdir(main_dir))
total = len(labels)
print(f"total images in ripe folder is {total}")
print(f"found {riped} riped which are {riped/total*100}%")
print(f"found {unriped} unriped which are {unriped/total*100}%")
print(f"found {semi} semi-riped which are {semi/total*100}%")
# %%
