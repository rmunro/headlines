import eel
import os
import random
import re
import time
import gzip
import csv
import hashlib
import datetime
import shutil
from random import shuffle
import torch
import torch.optim as optim

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
# downloading DistilBERT will take a while the first time

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

eel.init('./')

label = "" # current label we are annotating
annotation_count = 0 # current count of annotations since last training
last_annotation = 0 # time of most recent annotation
currently_training = False # if currently training
min_evaluation = 50

unlabeled_data = []
unlabeled_data_path = "headlines_v1.csv.gz"

labeled_urls = {} # track already-labeled data

predicted_confs = [] # predicted confidence of current label on most recent model (5th index)
all_predicted_confs = [] # all predicted confidences from a model (5th index)

verbose = True


def is_evaluation(headline):
    '''Returns true if the headline should be evaluation data
    
       Based on md5 hex value of headline to ensure consistency with repeated headlines
    '''
    hexval = hashlib.md5(headline.encode('utf-8')).hexdigest()
    intval = int(hexval, 16)

    if intval%4 == 0:
        return True
    else:
        return False
    


@eel.expose
def tag_as_interesting(item):
    '''Saves an item to the "interesting.csv" list
       
       These examples can be used to create detailed instructions later 
    '''
  
    global label
    filepath = "data/"+label+"/interesting.csv"
    append_data(filepath, [item])



@eel.expose
def add_annotation(item, positive=True):
    '''Record an annotation 
    
       The annotation is saved to file and the current model is 
       incrementally updated   
    '''
    global annotation_count
    global min_evaluation
    global current_model
    
    if verbose:
        print(label)
        print("adding annotation")
        print(item)
    
    headline = item[1]

    directory = "data/"+label+"/"
        
    if positive:
        labelpath = "positive.csv"
        labels = torch.tensor([1]).unsqueeze(0)  

    else:
        labelpath = "negative.csv" 
        labels = torch.tensor([0]).unsqueeze(0)   

    filepath = directory + labelpath

    if item[3].startswith("Random") and is_evaluation(item[1]):
        evaluation_filepath = directory + "evaluation_" + labelpath
        evaluation_items = load_headlines(evaluation_filepath)
        if len(evaluation_items) < min_evaluation:
            filepath = evaluation_filepath
    
    labeled_urls[item[2]] = True # record this is now labeled
    
    print("appending "+filepath)
    append_data(filepath, [item])

    annotation_count += 1
    global last_annotation
    last_annotation = time.time()  
    
    # incrementally update current model   
    inputs = tokenizer(headline, return_tensors="pt")
    train_item(current_model, inputs, labels)
         
 
 
@eel.expose
def get_existing_labels():
    '''Returns labels that were created previously              
    '''
    labels = []
    files = os.listdir('data') 
    for file_name in files:
        if os.path.isdir('data/'+file_name):
            labels.append(file_name)
    return labels
    

@eel.expose
def create_label(new_label):
    '''Creates a new label 
       
       If the label already exists, it makes that current
    '''
    global label    
    global annotation_count
    
    if not re.match("^[\w_\- ]+$",new_label):
        # invalid label name: only word characters, space, hyphen, and underscore are permitted
        return False
    if os.path.exists('data/'+new_label):
        label = new_label
        annotation_count = 1 # will trigger retraining
        load_existing_model()
        return True
    else:
        os.makedirs('data/'+new_label)
        label = new_label
        load_existing_model()
        return True


@eel.expose
def current_label():
    return label



@eel.expose
def get_data_to_annotate(num=1, filter="", year="", uncertainty=False):
    '''Returns unlabeled data to be annotated 
       
    Keyword arguments:
        num -- the number of items to return 
        filter -- return only headlines within this string
        year -- return only headlines from this year
        uncertainty -- return the most uncertain items from the model
        
    For uncertainty, the strategy prefers the most recent model, then 
    backs off to older models if there aren't enough items to return
    with the most recent model, given the other filters.
       
    '''
    
    global unlabeled_data
    global predicted_confs
    global all_predicted_confs
        
        

    if random.randint(0,99) == 0:
        # random re-shuffle every 100th annotation
        random.shuffle(unlabeled_data) 

    sample_data = unlabeled_data
        
    if uncertainty:
        if len(predicted_confs) == 0 and len(all_predicted_confs) == 0:
            # no predictions yet - choose via other filters
            if verbose:
                print("No model predictions yet")
            return get_data_to_annotate(num, filter, year, False)
        # try most recent model, then all models, then all items
        
        sample_data = predicted_confs + all_predicted_confs + unlabeled_data
    
    
    newdata = []
    for item in sample_data:
        date = item[0]
        item_year = date[:4]
        headline = item[1]        
        url = item[2]
        
        if url in labeled_urls:
            continue
            
        if len(headline) < 2:
            continue
        
        if filter != "" and filter not in headline:
            continue
        if year != "" and year != item_year:
            continue
                
        strategies = []
        if filter == "" and not uncertainty:
            strategies.append("Random")
        if uncertainty:
            strategies.append("Uncertain")
        if filter != "":
            strategies.append("Filter:"+filter)
        if year != "":
            strategies.append("_Year:"+year)
            
        sampling_strategy = "_".join(strategies)    
            
        while len(item) < 4:
            item.append(None)
            
        item[3] = sampling_strategy
            
        newdata.append(item)
        
        if len(newdata) == num:
            break
    
    return newdata



@eel.expose
def save_current_predictions():
    '''save predictions from all data
    
    Note that this can take some time! 
    
    '''
    global label 

    timestamp = re.sub('\.[0-9]*','_',str(datetime.datetime.now())).replace(" ", "_").replace("-", "").replace(":","")    
    savepath = "data/"+label+"/predictions_"+timestamp+".csv"     
    get_predictions(max_per_year = -1, savepath = savepath)
    


def append_data(filepath, data):
    with open(filepath, 'a', errors='replace') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    csvfile.close()


def load_headlines(filepath):
    '''Load existing labeled data    
    '''
    headlines = []
    if not os.path.exists(filepath):
        return []
    
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for item in reader:
            headlines.append(item[1])
            labeled_urls[item[2]] = True # record this is now labeled
            
    return headlines


def evaluate_item(model, inputs):
    '''Predict single item
    '''
    with torch.no_grad():
        outputs = model(**inputs)   
        logits = outputs[0][0]
        conf = torch.softmax(logits, dim=0)
        return conf


@eel.expose
def evaluate_model(model):
    '''Evaluate model against current evaluation data
    '''
    global label 
    print("evaluating model")
    directory = "data/"+label+"/"

    positive_filepath = directory + "evaluation_positive.csv"    
    positive_items = load_headlines(positive_filepath)
      
    correct = 0
    incorrect = 0
    
    total_conf = 0.0
    
    tp = 0
    fp = 0
    fn = 0        
               
    print(len(positive_items))
    for headline in positive_items:
        inputs = tokenizer(headline, return_tensors="pt")
        conf = evaluate_item(model, inputs)
        total_conf += 1.0-conf
        if conf[0] < 0.5: 
            correct += 1 
            tp += 1           
        else:
            incorrect += 1
            fn += 1

    negative_filepath = directory + "evaluation_negative.csv"    
    negative_items = load_headlines(negative_filepath)
        
    for headline in negative_items:
        inputs = tokenizer(headline, return_tensors="pt")
        conf = evaluate_item(model, inputs)
        total_conf += conf
        if conf[0] > 0.5: 
            correct += 1
        else:
            incorrect += 1
            fp += 1

    if tp == 0:
        precision = 0.0
        recall = 0.0
    else:
        precision = (float(tp)/float(tp+fp))
        recall = (float(tp)/float(tp+fn))
    if precision == 0 or recall == 0:
        fscore = 0.0
    else:
        fscore = (2 * precision * recall) / (precision + recall)

    if verbose:
        print("accuracy: "+str(correct/(correct+incorrect)))
        print("precision: "+str(precision))
        print("recall: "+str(recall))
        print("fscore: "+str(fscore))
        print("ave conf: "+str(total_conf[0]/(correct+incorrect)))
        
    return(fscore)



def train_item(model, inputs, labels):
    '''Update model with single item
    '''
    model.zero_grad() 
    outputs = model(**inputs, labels=labels)                
    loss, logits = outputs[:2]   
    
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss.backward()
    optimizer.step()     



def retrain(filepath, epochs_per_item=2, min_to_train=10):
    '''Retrain a new model from scratch 
    '''
    global label
    global current_model
    global currently_training

    if currently_training:
        "skipping while model already training"
        return

    positives = load_headlines(filepath+"positive.csv")
    negatives = load_headlines(filepath+"negative.csv")

    if len(positives) < min_to_train or len(negatives) < min_to_train:
        print("too few annotations to train: less than "+str(min_to_train))
        return

    
    currently_training = True
    
    new_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

    
    # sample each item no more than `epochs_per_item` for least frequent label
    iterations = int(min(len(positives), len(negatives)) * epochs_per_item)
    
        
    for i in range(0, iterations):
        positive_headline = random.choice(positives)    
        positive_inputs = tokenizer(positive_headline, return_tensors="pt")
        positive_labels = torch.tensor([1]).unsqueeze(0)  
 
        train_item(new_model, positive_inputs, positive_labels)
         
        negative_headline = random.choice(negatives)        
        negative_inputs = tokenizer(negative_headline, return_tensors="pt")
        negative_labels = torch.tensor([0]).unsqueeze(0)  
                   
        train_item(new_model, negative_inputs, negative_labels)

        eel.sleep(0.01) # allow other processes through
 
    new_fscore = evaluate_model(new_model)
    current_fscore = evaluate_model(current_model)
      
    if(new_fscore > current_fscore):
        print("replacing model!")
        current_model = new_model
        timestamp = re.sub('\.[0-9]*','_',str(datetime.datetime.now())).replace(" ", "_").replace("-", "").replace(":","")
        accuracy = str(round(new_fscore, 4))
                     
        model_path = "data/"+label+"/"+timestamp+accuracy+".model"
        current_model.save_pretrained(model_path)
        if verbose:
            print("saved model to "+model_path)
        clean_old_models()
            
        get_predictions()
    else:
        print("staying with old model")
        
    currently_training = False


def clean_old_models(max_prior=4):
     global label
     models = []
     
     files = os.listdir('data/'+label) 
     for file_name in files:
         if os.path.isdir('data/'+label+'/'+file_name):
             if file_name.endswith(".model"):
                 models.append('data/'+label+"/"+file_name)
    
     if len(models) > max_prior:
         for filepath in models[:-4]:
             assert("data" in filepath and ".model" in filepath and label in filepath and len(label) > 0)
             if verbose:
                 print("removing old model "+filepath)
             shutil.rmtree(filepath)
    


def load_existing_model():
    global label
    global current_model 

    model_path = ""
    
    files = os.listdir('data/'+label) 
    for file_name in files:
        if file_name.endswith(".model"):
            model_path = 'data/'+label+"/"+file_name
                
    if model_path != '':    
        if verbose:
            print("Loading model from "+model_path)
        current_model = DistilBertForSequenceClassification.from_pretrained(model_path)
        eel.sleep(0.1)
    else:
        if verbose:
            print("Creating new uninitialized model (OK to ignore warnings)")

        current_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')



def get_predictions(max_per_year = 1000, earliest_year="2011", savepath=""):
    '''Get predictions from the current model 
       
    Keyword arguments:
        max_per_year -- the maximum number for each year (-1 is no limit)
        earliest_year -- the earliest year to include in results
        savepath -- save predictions to file (does not save if empty string)
               
    '''
    global labeled_urls
    global unlabeled_data
    global current_model
    global predicted_confs
    global all_predited_confs
    
    if verbose:
        print("Getting predictions")
        if savepath != '':
            print("Saving to "+savepath)
    
    new_predicted_confs = []
    
    count = 0
    count_by_year = {}
    
    shuffle(unlabeled_data)
    
    for item in unlabeled_data:
        url = item[2]
        headline = item[1]
        date = item[0]
        year = date[:4]
        
        if int(year) < int(earliest_year):
            continue
        
        if year not in count_by_year:
            count_by_year[year] = 1
        elif count_by_year[year] >= max_per_year and max_per_year != -1:
            continue
        else:
            count_by_year[year] += 1
        
        if url in labeled_urls:
            continue 
            
        inputs = tokenizer(headline, return_tensors="pt")
        conf = evaluate_item(current_model, inputs)
        real_conf = conf.data[1]
        
        least_conf = 2 * (1 - max(real_conf, 1-real_conf))
        
        while(len(item) < 6):
            item.append(None)
        
        item[4] = real_conf
        item[5] = least_conf
        
        new_predicted_confs.append(item)
        all_predicted_confs.append(item)
        
        if savepath != "":
            append_data(savepath, [item])
        
        count+=1
        if count%50 == 0:
            eel.sleep(0.01) # allow other processes through
            if verbose:
                print(count)
        
    predicted_confs = new_predicted_confs # update most recent

    predicted_confs.sort(reverse=True, key=lambda x: x[5]) 
    all_predicted_confs.sort(reverse=True, key=lambda x: x[5])  
    
    return new_predicted_confs


@eel.expose
def get_positives_per_year(num=10, threshold=0.6, earliest_year="2011"):
    '''Returns confident predictions for the given years
       
    Keyword arguments:
        num -- the number of items to return per year
        threshold -- return only those items above this confidence
        earliest_year -- the earliest year to include in results
              
    '''
    global all_predicted_confs
    print(len(all_predicted_confs))
    
    confident_items_by_year = {}
    items_to_return = {}
    
    items = all_predicted_confs.copy()
    shuffle(items)
    
    for item in items:
        conf = item[4]
        
        if conf >= threshold:
            date = item[0]
            year = date[:4]
            
            if int(year) < int(earliest_year):
                continue
            
            if year not in confident_items_by_year:
                confident_items_by_year[year] = [item]
            else:
                confident_items_by_year[year].append(item)

        
    for year in confident_items_by_year:
        items = confident_items_by_year[year]
        shuffle(items)
        items_to_return[year] = items[:num]
        
    print(items_to_return)
    return items_to_return
    
        


def check_to_train():
    '''Continually check to try retraining entire model from scratch
    '''
    global annotation_count
    global last_annotation
    global currently_training
    global label
    while True:
        print("Checking to retrain "+str(label))
        
        ct = time.time()       
        if currently_training or annotation_count == 0 or ct - last_annotation < 20:
            eel.sleep(10)   
            continue
        
        
        annotation_count = 0 # reset counter for annotations
        
        retrain('data/'+label+'/')
        
        # print(ct)
       
        eel.sleep(10)                  # Use eel.sleep(), not time.sleep()

eel.spawn(check_to_train)



# LOAD ALL THE DATA TO GET STARTED ! 

unlabeled_file = gzip.open(unlabeled_data_path, mode='rt')
csvobj = csv.reader(unlabeled_file,delimiter = ',',quotechar='"')
for row in csvobj:
    unlabeled_data.append(row)
random.shuffle(unlabeled_data)

eel.start('headlines.html', size=(800, 600))



