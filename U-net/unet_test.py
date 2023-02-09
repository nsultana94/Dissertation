import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
import sklearn.metrics as skm

from sklearn.metrics import jaccard_score, accuracy_score

DATA_URL = "/cs/student/projects1/2019/nsultana/"

trainset = SegmentationDataset(train_df, get_train_augs())

validset = SegmentationDataset(valid_df, get_valid_augs())

testset = SegmentationDataset(test_df, get_test_augs())

from torch.utils.data import DataLoader

trainloader = DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True) #every epoch batches shuffles
validloader = DataLoader(validset, batch_size = BATCH_SIZE, shuffle = True)


model = SegmentationModel()
model.to(DEVICE); #i.e CUDA

def initialiseDictionary():
  labels = [0,1,2,3,4,5,6,7]
  label_stats = {}
  for label in labels:
    label_stats[label] = {'tp': 0, 'fn': 0, 'fp': 0}
  return label_stats


model.load_state_dict(torch.load('/content/drive/My Drive/Dissertation/batchsize.pt'))
stats =initialiseDictionary()

labels = [0,1,2,3,4,5,6,7]
for idx in range (0, len(validset)):
  
  image, mask = validset[idx]
  model.eval()
  logits_mask = model(image.to(DEVICE).unsqueeze(0)) # (c, h, w ) -> (1, c, h , w)
  predictions = torch.nn.functional.softmax(logits_mask, dim=1)
  pred_labels = torch.argmax(predictions, dim=1)

  prediction = pred_labels.to('cpu').flatten().numpy()
  ground_truth = mask.to('cpu').flatten().numpy()


  #tp, fp, fn, tn = smp.metrics.get_stats(prediction, ground_truth, mode='multiclass', num_classes = 8)

  conf_matrix = skm.multilabel_confusion_matrix(ground_truth, prediction,labels=labels)
  for label in labels:
    stats[label]['tp'] += conf_matrix[label][1][1] 
    stats[label]['fn'] += conf_matrix[label][1][0] 
    stats[label]['fp'] += conf_matrix[label][0][1] 

for label in labels:
    tp = stats[label]['tp'] 
    fn = stats[label]['fn'] 
    fp = stats[label]['fp'] 
    iou = tp / ( fp + tp + fn)
    print(f"class {label} iou: {iou}")
