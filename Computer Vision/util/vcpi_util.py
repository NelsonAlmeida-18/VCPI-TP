from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix

import torch
import torchvision
import os
import datetime
import time
from tqdm import tqdm



def show_history_plus(history, fields):

    # summarize history for accuracy
    for hist in fields:

        plt.plot(history[hist])

        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
    plt.legend(list(history.keys()), loc='lower right')
    plt.show()



def show_history(history):

    for key in history.keys():
    
        # summarize history for accuracy
        plt.plot(history['accuracy'])
        plt.title(key)
        plt.ylabel(key)
        plt.xlabel('epoch')
        plt.show()


def show_histories(histories, model_names, field='accuracy'):

    # summarize history for accuracy
    for hist in histories:
        plt.plot(hist[field])

    plt.ylabel(field)
    plt.xlabel('epoch')
    plt.legend(model_names, loc='lower right')
    plt.show()


def show_accuracies(train_acc, eval_acc, model_names): 
    fig, ax = plt.subplots()
    X = np.arange(len(model_names))

    minT = min(train_acc)
    minE = min(eval_acc)
    min_val = min([minT, minE])
    
    plt.bar(X, eval_acc, width = 0.4, color = 'b', label='eval')
    plt.bar(X + 0.4, train_acc, color = 'r', width = 0.4, label = "train")
    plt.xticks(X + 0.4 / 2, model_names)
    plt.ylim(top = 100, bottom = min_val - 2)
    plt.legend(loc='lower right')
    plt.show()    


def show_histogram(data, classes):

    target_np = [x.numpy().item() for x in data]
    res = Counter(target_np)
    print(res)

    values = [res[x] for x in range(len(classes))]
    indexes = np.arange(len(classes))


    plt.bar(indexes, values, 1)
    plt.xticks(indexes , classes, rotation=45, ha='right')
    plt.show()   


def show_confusion_matrix(ground_truth, preds, num_classes):    

    cf_matrix = confusion_matrix(ground_truth, preds)


    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], range(num_classes), range(num_classes))
    plt.figure(figsize=(12,6))

    sn.heatmap(df_cm, annot=True, annot_kws={"size": 10} , fmt='.3f') # font size

    plt.show()
    

def show_loaded_images(rows, cols, data,classes):

    width= 2 * rows
    height= 2 * cols

    f, axes= plt.subplots(rows,cols,figsize=(height,width))
    fig=plt.figure()

    for a in range(rows*cols):
        img, label = data[a]
        subplot_title=(classes[label])
        axes.ravel()[a].set_title(subplot_title)  
        axes.ravel()[a].imshow(np.transpose(img.numpy(), (1,2,0)), cmap=plt.cm.gray)
        axes.ravel()[a].axis('off')
    fig.tight_layout()    
    plt.show() 


def show_transformed_images(rows, cols, data, classes):

    width= 2 * rows
    height= 2 * cols

    f, axes= plt.subplots(rows,cols,figsize=(height,width))
    fig=plt.figure()

    for a in range(rows*cols):
        img, target = data[a]
        subplot_title=(classes[target])
        axes.ravel()[a].set_title(subplot_title)  
        axes.ravel()[a].imshow(np.transpose(img.numpy(),(1,2,0)), cmap=plt.cm.gray)
        axes.ravel()[a].axis('off')
    fig.tight_layout()    
    plt.show()         

def show_predicted_images(rows, cols, data, classes):

    width= 2 * rows
    height= 2 * cols

    f, axes= plt.subplots(rows,cols,figsize=(height,width))
    fig=plt.figure()

    for a in range(rows*cols):
        img = data[a]
        axes.ravel()[a].imshow(np.transpose(img.numpy(),(1,2,0)), cmap=plt.cm.gray)
        axes.ravel()[a].axis('off')
    fig.tight_layout()    
    plt.show()         

def show_images(rows, cols, images, targets, classes):

    width= 2 * rows
    height= 2 * cols

    f, axes= plt.subplots(rows,cols,figsize=(height,width))
    fig=plt.figure()

    for a in range(rows*cols):
        img, target = images[a], targets[a]
        subplot_title=(classes[target])
        axes.ravel()[a].set_title(subplot_title)  
        axes.ravel()[a].imshow(np.transpose(img.numpy(),(1,2,0)), cmap=plt.cm.gray)
        axes.ravel()[a].axis('off')
    fig.tight_layout()    
    plt.show()      


def plot_image(i, predictions_array, true_label, img, classes):

    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])  
    img_np = np.transpose(img.numpy(), (1,2,0))
    plt.imshow(img_np, cmap=plt.cm.gray)  
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
      color = 'blue'
    else:
      color = 'red' 
    plt.xlabel("{} {:2.0f}% ({})".format(classes[predicted_label],
                                  100*np.max(predictions_array),
                                  classes[true_label]),
                                  color=color)

def plot_value_array(i, predictions_array, true_label, num_classes):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(num_classes))
    plt.yticks([])
    thisplot = plt.bar(range(num_classes), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)  
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.

def plot_predictions(images, predictions, ground_truth, classes, num_rows= 5, num_cols=3 ):

    num_images = min(num_rows*num_cols, len(predictions))
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions[i], ground_truth, images, classes)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions[i], ground_truth, len(classes))
    plt.tight_layout()
    plt.show()

def build_confusion_matrix(model, dataset, test_set, device):

    preds = []
    ground_truth = []

    for images, targets in dataset:

        predictions = model(images.to(device))
        preds_sparse = [np.argmax(x) for x in predictions.cpu().detach().numpy()]
        preds.extend(preds_sparse)
        ground_truth.extend(targets.numpy())

    show_confusion_matrix(ground_truth, preds, len(test_set.classes))      

    # De forma a estabilizar os resultados, quando a accuracy não melhora ao fim de x épocas, atualizamos a taxa de aprendizagem para uma mais baixa
# Ao pararmos quando piora/não melhora convém guardarmos a iteração anterior visto esta ser a melhor
def train(device, model, training_data,  data_loader, val_sub, val_sub_loader, epochs, loss_fn, optimizer, scheduler, early_stopping, save_prefix="model", path_name = ""):

    if "models" not in os.listdir():
      os.mkdir("models")

    #Lets create the path to save the model  
    if path_name not in os.listdir("models"):
      date = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
      if path_name == "": path_name = f"model_{date}"
      os.mkdir(f"models/{path_name}")

    #Lets create the logs file
    logfile = open(f"models/{path_name}/logs_{path_name}.txt", "a")

    model.train()

    history = {}
    history['accuracy'] = []
    history['val_accuracy'] = []
    history['val_loss'] = []
    history['loss'] = []
    best_val_loss = np.inf

    start_time = time.time()
    stop_time = time.time()
    accuracy = 0
    val_loss = 0
    val_accuracy = 0



    for epoch in tqdm(range(epochs), desc=f"Epoch: {epoch:03d}; Acc = {accuracy:0.4f}; Val_loss = {val_loss}; Vall_acc: {val_accuracy}; Time: {(stop_time-start_time):0.4f}"):
        # O droput durante o treino deita fora alguns neurónios, durante a avaliação usa sempre os neurónios todos
        # Batch normalization durante o treino usa a média e o desvio padrão da batch que recebeu, mas durante a avaliação usa a média que vai calculando durante o treino
        #  Todos os outros layers são iguais durante o treino e avaliação
        model.train()
        start_time = time.time()
        correct = 0
        running_loss = 0

        for i, (inputs,targets) in enumerate(data_loader,0):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            loss = loss_fn(outputs, targets)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            _,pred = torch.max(outputs, 1)
            correct += (pred == targets).sum()
            running_loss += loss

        
        model.eval()
        t_correct = 0
        val_loss = 0
        # torch.no_grad() é usado para desativar o cálculo do gradiente, o que acelera o processo e reduz a memória
        with torch.no_grad():
            for i,t in val_sub_loader:
                i = i.to(device)
                t = t.to(device)
                o = model(i)
                _,p = torch.max(o, 1)
                t_correct += (t == p).sum()

                # Loss entre os targets e os outputs
                val_loss += loss_fn(o,t)

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']

        if old_lr != new_lr:
            #print(f'==> Learning rate changed from {old_lr} to {new_lr}')
            #Lets write the changes to the log file
            logtime = datetime.datetime.now().strftime("%H_%M_%S")
            logfile.write(f"[{logtime}] -> Learning rate changed from {old_lr} to {new_lr} in {epoch} epoch\n")
        

        stop_time = time.time()
        accuracy = 100 * correct/len(training_data)
        val_accuracy = 100*t_correct/len(val_sub)
        
        #print(f'Epoch: {epoch:03d}; Acc = {accuracy:0.4f}; Val_loss = {val_loss}; Vall_acc: {val_accuracy}; Time: {(stop_time- start_time):0.4f}')
        history["val_accuracy"].append(val_accuracy.cpu().numpy())
        history['accuracy'].append(accuracy.cpu().numpy())
        history['loss'].append(running_loss.cpu().detach().numpy())
        

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "accuracy": accuracy,
                "val_accuracy": val_accuracy,
                "loss": running_loss,
                "val_loss": val_loss,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict()
                },
                f"models/{path_name}/{save_prefix}_{epoch}.pth"
            )
            #print("==> Model saved")
            #Lets write to the log file
            logtime = datetime.datetime.now().strftime("%H_%M_%S")
            logfile.write(f"[{logtime}] -> Model saved with name {save_prefix}_{epoch} in {epoch} epoch with validation loss={val_loss}\n")
        
        if early_stopping(val_loss):
            #print("==> Early stopping")
            #Lets write to the log file
            logtime = datetime.datetime.now().strftime("%H_%M_%S")
            logfile.write(f"[{logtime}] -> Training stopped early with {val_loss} validation loss in epoch {epoch}\n")
            break

    #Lets write to the log file
    logtime = datetime.datetime.now().strftime("%H_%M_%S")
    logfile.write(f"[{logtime}] -> Training finished with {val_loss} loss {accuracy} accuracy with training data and {val_accuracy} accuracy in the validation data in epoch {epoch}\n")
    endtime = time.time()
    time_elapsed = endtime - start_time
    print(f"==> Training finished in {time_elapsed} minutes")
    print(f"Training stats: {accuracy} accuracy | {val_accuracy} validation accuracy | {val_loss} loss")
    return(history)

def evaluate(model, data_loader, device):

    # sets the model in evaluation mode.
    # although our model does not have layers which behave differently during training and evaluation
    # this is a good practice as the models architecture may change in the future
    model.eval()

    correct = 0
    
    for _, (images, targets) in enumerate(data_loader):
         
        # forward pass, compute the output of the model for the current batch
        outputs = model(images.to(device))

        # "max" returns a namedtuple (values, indices) where values is the maximum 
        # value of each row of the input tensor in the given dimension dim; 
        # indices is the index location of each maximum value found (argmax).
        # the argmax effectively provides the predicted class number        
        _, preds = torch.max(outputs, dim=1)

        correct += (preds.cpu() == targets).sum()

    return (correct / len(data_loader.dataset)).item()


class Early_Stopping():

    def __init__(self, patience=3, min_delta=0.000001):
        self.patience = patience
        self.min_delta = min_delta

        self.min_loss = np.inf
        self.counter = 0

    def __call__(self, loss):
        if (loss+self.min_delta < self.min_loss):
            self.min_loss = loss
            self.counter = 0

        else:
            self.counter += 1
            if self.counter > self.patience:
                return True

        return False
    
