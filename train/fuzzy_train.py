import numpy as np
import pandas as pd
import os
from fastai import *
from fastai.vision import *
import torch.nn as nn

# PATH is used to store the path of the training set
PATH = Path('***')
# PATH1 is used to store test set addresses
PATH1 = Path('***')
# Model_Path is used to store model addresses(such as vgg,resnet alexnet and so on)
Model_Path = Path('***')


transform = get_transforms(max_rotate=7.5,
                           max_zoom=1.15,
                           max_lighting=0.15,
                           max_warp=0.15,
                           p_affine=0.8,
                           p_lighting = 0.8,
                           xtra_tfms= [
                               pad(mode='zeros'),
                               symmetric_warp(magnitude=(-0.1,0.1)),
                               cutout(n_holes=(1,6),length=(5,20))])
data = ImageDataBunch.from_folder(PATH, train="train/",
#                                  valid="train/",
                                  test="test/",
                                  valid_pct=.2,
                                  ds_tfms=transform,
                                  size=224,bs=32,
                                  ).normalize(imagenet_stats)
print(data)
learn = cnn_learner(data, models.vgg19_bn, pretrained=False, metrics=[error_rate, FBeta(average='weighted')], wd=1e-1, callback_fns=ShowGraph)
learn.model_dir = Model_Path
learn.fit_one_cycle(38)
learn.unfreeze()
learn.fit_one_cycle(8, max_lr=slice(1e-6,3e-4))
learn.save('train_vgg19_bn_1_origin_2')

#stage 4 test
transform = get_transforms()
data_test =  ImageDataBunch.from_folder(PATH1,
                                  valid_pct=0,
#                                  ds_tfms=transform,
                                  size=224,bs=32,
                                  ).normalize(imagenet_stats)

print(data_test)

learn1 = cnn_learner(data_test, models.vgg19_bn, pretrained=False, metrics=[error_rate, FBeta(average='weighted')], wd=1e-1, callback_fns=ShowGraph)
learn1.model_dir = Model_Path
learn1.load('train_vgg19_bn_1_origin_2')
res=learn1.get_preds(data_test.train_dl)
preds=[]
for item in res[0].tolist():
  maxnum=max(item)
  if(item[0]==maxnum):
    preds.append(0)
  elif(item[1]==maxnum):
    preds.append(1)
  elif(item[2]==maxnum):
    preds.append(2)
  else:
    preds.append(3)
labels=res[1].tolist()
from sklearn.metrics import classification_report, confusion_matrix
x=confusion_matrix(labels, preds)
print(x)
print(classification_report(labels, preds))

# 2 calculate TP/TN/FP/FN
FP = x.sum(axis=0) - np.diag(x)
FN = x.sum(axis=1) - np.diag(x)
TP = np.diag(x)
TN = x.sum() - (FP + FN + TP)
FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)
# print(TP)
# print(TN)
# print(FP)
# print(FN)

# 3 calculate others
TPR = TP / (TP + FN)  # Sensitivity/ hit rate/ recall/ true positive rate
TNR = TN / (TN + FP)  # Specificity/ true negative rate
PPV = TP / (TP + FP)  # Precision/ positive predictive value
NPV = TN / (TN + FN)  # Negative predictive value
FPR = FP / (FP + TN)  # Fall out/ false positive rate
FNR = FN / (TP + FN)  # False negative rate
FDR = FP / (TP + FP)  # False discovery rate
ACC = TP / (TP + FN)  # accuracy of each class
# print(TPR)
# print(TNR)
# print(PPV)
# print(NPV)
# print(FPR)
# print(FNR)
# print(FDR)
# print(ACC)
ACC_micro = (sum(TP) + sum(TN)) / (sum(TP) + sum(FP) + sum(FN) + sum(TN))
ACC_macro = np.mean(ACC) # to get a sense of effectiveness of our method on the small classes we computed this average (macro-average)
F1 = (2 * PPV * TPR) / (PPV + TPR)
F1_macro = np.mean(F1)

MCC=(TP*TN-FP*FN)/np.sqrt((FP+TP)*(TP+FN)*(TN+FP)*(TN+FN))
MCC_macro=np.mean(MCC)


print(ACC_micro)
print(np.mean(TPR))
print(np.mean(TNR))
print(np.mean(PPV))
print(F1_macro)
print(MCC_macro)