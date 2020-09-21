import eel, os, random, sys, re
import time
import gzip
import csv
import hashlib
from random import shuffle


'''
import active_learning
from active_learning import * 
'''

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
# downloading DistilBERT will take a while the first time

import torch
import torch.optim as optim




tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
current_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')


eel.init('./')

label = "" # current label we are annotating
annotation_count = 0 # current count of annotations since last training
last_annotation = 0 # time of most recent annotation
currently_training = False # if currently training
min_evaluation = 50

unlabeled_data = []
unlabeled_data_path = "headlines_v1.csv.gz"

labeled_urls = {} # track already-labeled data

predicted_confs = {} # predicted confidence of current label on current model by url
all_predicted_confs = {} # most recent predicted confidence, not necessarily from most recent model

verbose = True

@eel.expose
def pick_file(folder):
    if os.path.isdir(folder):
        return random.choice(os.listdir(folder))
    else:
        return 'Not valid folder'


def is_evaluation(headline):
    hexval = hashlib.md5(headline.encode('utf-8')).hexdigest()
    intval = int(hexval, 16)

    if intval%4 == 0:
        return True
    else:
        return False
    


@eel.expose
def add_annotation(item, positive=True, evaluation=False):
    global annotation_count
    global min_evaluation
    global current_model
    print(label)
    
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
def create_label(new_label):
    global label    
    global annotation_count
    
    if not re.match("^[\w_\- ]+$",new_label):
        # invalid label name: only word characters, space, hyphen, and underscore are permitted
        return False
    if os.path.exists('data/'+new_label):
        label = new_label
        annotation_count = 1 # will trigger retraining
        return True
    else:
        os.makedirs('data/'+new_label)
        label = new_label
        print(label)
        return True


@eel.expose
def current_label():
    return label


@eel.expose
def pr():
    time.sleep(10)
    print("done")
    print(time.time)


@eel.expose
def get_data_to_annotate(num, filter="", year=""):
    global unlabeled_data
    random.shuffle(unlabeled_data)
    
    print(filter)
    print(year)
    
    print("selecting from "+str(len(unlabeled_data)))
    
    newdata = []
    for item in unlabeled_data:
        date = item[0]
        item_year = date[:4]
        headline = item[1]
        is_evaluation(headline)
        
        url = item[2]
        if url in labeled_urls:
            continue
        
        if filter != "" and filter not in headline:
            continue
        if year != "" and year != item_year:
            continue
        
        if filter == "":
            sampling_strategy = "Random"
        else:
            sampling_strategy = "Filter:"+filter
        if year != "":
            sampling_strategy += "_Year:"+year
            
        item.append(sampling_strategy)
            
        newdata.append(item)
        if len(newdata) == num:
            break
    
    return newdata


def append_data(filepath, data):
    with open(filepath, 'a', errors='replace') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    csvfile.close()


def load_headlines(filepath):
    # FOR ALREADY LABELED ONLY
    # csv format: [DATE, TEXT, URL,...]
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
    with torch.no_grad():
        outputs = model(**inputs) #, labels=labels)   
        logits = outputs[0][0]
        conf = torch.softmax(logits, dim=0)
        # print(conf)
        return conf


@eel.expose
def evaluate_model(model):
    global label 
    print("evaluating model")
    directory = "data/"+label+"/"

    positive_filepath = directory + "evaluation_positive.csv"    
    # positive_filepath = directory + "positive.csv"    
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
        if conf[0] < 0.5: # CHECK
            # print("correct "+str(conf)+" "+  headline)
            correct += 1 
            tp += 1           
        else:
            # print("incorrect "+str(conf)+" "+  headline)
            incorrect += 1
            fn += 1

    negative_filepath = directory + "evaluation_negative.csv"    
    #negative_filepath = directory + "negative.csv"    
    negative_items = load_headlines(negative_filepath)
        
    for headline in negative_items:
        inputs = tokenizer(headline, return_tensors="pt")
        conf = evaluate_item(model, inputs)
        total_conf += conf
        if conf[0] > 0.5: # CHECK
            # print("correct "+str(conf)+" "+ headline)
            correct += 1
        else:
            # print("incorrect "+str(conf)+" "+ headline)
            incorrect += 1
            fp += 1

    precision = (float(tp)/float(tp+fp+0.00001))
    recall = (float(tp)/float(tp+fn+0.00001))
    if precision == 0 or recall == 0:
        fscore = 0.0
    else:
        fscore = (2 * precision * recall) / (precision + recall)

    
    print("accuracy: "+str(correct/(correct+incorrect)))
    print("precision: "+str(precision))
    print("recall: "+str(recall))
    print("fscore: "+str(fscore))
    
    
    print("ave conf: "+str(total_conf[0]/(correct+incorrect)))
    return(fscore)



def train_item(model, inputs, labels):
    model.zero_grad() 
    outputs = model(**inputs, labels=labels)                
    loss, logits = outputs[:2]   
    
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss.backward()
    optimizer.step()     



def retrain(filepath, epochs_per_item=2, min_to_train=10):
    global current_model
    global currently_training

    if currently_training:
        "skipping while model already training"
    
    currently_training = True
    
    new_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

    positives = load_headlines(filepath+"positive.csv")
    negatives = load_headlines(filepath+"negative.csv")
    
    if len(positives) < min_to_train or len(negatives) < min_to_train:
        print("too few annotations to train")
        return
    
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
        # print("."+str(i)+" of "+str(iterations))
        eel.sleep(0.01) # allow other processes through
 
    new_fscore = evaluate_model(new_model)
    current_fscore = evaluate_model(current_model)
      
    if(new_fscore > current_fscore):
        print("replacing model!")
        current_model = new_model
        get_uncertainty()
    else:
        print("staying with old model")
        
    currently_training = False


def get_uncertainty(max_per_year = 1000, earliest_year="2011"):
    global labeled_urls
    global unlabeled_data
    global current_model
    global predicted_confs
    global all_predited_confs
    
    print("Getting predictions")
    
    
    new_predicted_confs = {}
    
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
        elif count_by_year[year] >= max_per_year:
            continue
        else:
            count_by_year[year] += 1
        
        if url in labeled_urls:
            continue 
            
        inputs = tokenizer(headline, return_tensors="pt")
        conf = evaluate_item(current_model, inputs)
        real_conf = conf.data[1]

        new_predicted_confs[url] = real_conf
        all_predicted_confs[url] = [real_conf, item]
        
        count+=1
        if count%50 == 0:
            eel.sleep(0.01) # allow other processes through
            print(count)
        
    predicted_confs = new_predicted_confs # update most recent



@eel.expose
def get_positives_per_year(num=10, threshold=0.6, earliest_year="2011"):
    global all_predicted_confs
    print(len(all_predicted_confs))
    
    confident_items_by_year = {}
    items_to_return = {}

    for url in all_predicted_confs:
        conf, item = all_predicted_confs[url]
        
        if conf >= threshold:
            date = item[0]
            year = date[:4]
            
            if int(year) < int(earliest_year):
            	continue
            
            if year not in confident_items_by_year:
                confident_items_by_year[year] = [item]
            else:
                confident_items_by_year[year].append(item)
            #print("conf good")
        # else:
            # print("conf too low")
        
    for year in confident_items_by_year:
        items = confident_items_by_year[year]
        shuffle(items)
        items_to_return[year] = items[:num]
        
    print(items_to_return)
    return items_to_return
    
        


def check_to_train():
    global annotation_count
    global last_annotation
    global currently_training
    global label
    while True:
        print("Checking to retrain "+str(label))
        
        ct = time.time()       
        if currently_training or annotation_count == 0 or ct - last_annotation < 20:
            # print("No new annotations or annotation within 20 seconds")
            # print(ct - last_annotation)
            eel.sleep(10)   
            continue
            # more than 20 seconds since last annotation was added
        
        # (re)retrain model! 
        
        annotation_count = 0 # reset counter for annotations
        
        retrain('data/'+label+'/')
        
        # print(ct)
       
        eel.sleep(10)                  # Use eel.sleep(), not time.sleep()

eel.spawn(check_to_train)



# directories with data
unlabeled_file = gzip.open(unlabeled_data_path, mode='rt')
csvobj = csv.reader(unlabeled_file,delimiter = ',',quotechar='"')
for row in csvobj:
    unlabeled_data.append(row)
    # print(row[0])

eel.start('headlines.html', size=(800, 600))




# create_label("Disaster Related")
# retrain("data/Disaster Related/")

