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

def train(model, train_loader, val_loader, epochs, loss_fn, optimizer,
           scheduler, early_stopper, device, save_prefix = 'model'):

    if not os.path.exists('models'):
        os.makedirs('models')
    
    if not os.path.exists(f'models/{save_prefix}'):
        os.makedirs(f'models/{save_prefix}')

    history = {}
    history['accuracy'] = []
    history['val_acc'] = []
    history['val_loss'] = []
    history['loss'] = []
    best_val_loss = np.inf

    for epoch in range(epochs):  # loop over the dataset multiple times

        model.train()
        start_time = time.time() 
        correct = 0
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader, 0):
            
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            loss = loss_fn(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss
            correct += (predicted == targets).sum()

        model.eval()
        v_correct = 0
        val_loss = 0.0
        with torch.no_grad():
            for i,t in val_loader:
                i = i.to(device)
                t = t.to(device)
                o = model(i)
                _,p = torch.max(o,1)
                
                #with torch.no_grad():
                val_loss += loss_fn(o, t)

                v_correct += (p == t).sum()

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        if old_lr != new_lr:
            print('==> Learning rate updated: ', old_lr, ' -> ', new_lr)

        epoch_loss = running_loss / len(train_loader.dataset)
        accuracy = 100 * correct / len(train_loader.dataset)
        v_accuracy = 100 * v_correct / len(val_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        stop_time = time.time()
        print(f'Epoch: {epoch:03d}; Loss: {epoch_loss:0.6f}; Accuracy: {accuracy:0.4f}; Val Loss: {val_loss:0.6f}; Val Acc: {v_accuracy:0.4f}; Elapsed time: {(stop_time - start_time):0.4f}')
        history['accuracy'].append(accuracy.cpu().numpy())
        history['val_acc'].append(v_accuracy.cpu().numpy())
        history['val_loss'].append(val_loss.cpu().detach().numpy())
        history['loss'].append(epoch_loss.cpu().detach().numpy())
 
        ###### Saving ######
        if val_loss < best_val_loss :
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model':model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'history': history,
                },
                f'models/{save_prefix}.pt')

        if early_stopper(val_loss):
            print('Early stopping!')
            break
        
    print('Finished Training')

    return(history)


def evaluate(model, data_loader, device):

    # sets the model in evaluation mode.
    # although our model does not have layers which behave differently during training and evaluation
    # this is a good practice as the models architecture may change in the future
    model.eval()

    correct = 0
    
    for i, (images, targets) in enumerate(data_loader):
         
        # forward pass, compute the output of the model for the current batch
        outputs = model(images.to(device))

        # "max" returns a namedtuple (values, indices) where values is the maximum 
        # value of each row of the input tensor in the given dimension dim; 
        # indices is the index location of each maximum value found (argmax).
        # the argmax effectively provides the predicted class number        
        _, preds = torch.max(outputs, dim=1)

        correct += (preds.cpu() == targets).sum()

    return (correct / len(data_loader.dataset)).item()



def show_bad_preds(model, dataset, classes, device, batch_size, test_set):
    k = 0
    iters = 0

    preds = []
    ground_truth = []
    imgs = torch.Tensor(15, 3, 32,32)

    iterator = iter(dataset)

    max_iters = test_set.__len__() / batch_size
    while k < 15 and iters < max_iters:

        images, targets = next(iterator)
        #print(images[0].shape)
        logits = model(images.to(device))
        
        #print(predictions[0])
        predictions = torch.nn.functional.softmax(logits, dim=1).cpu().detach().numpy()
        for i in range(len(predictions)):

            if np.argmax(predictions[i]) != targets[i] and k < 15:

                preds.append(predictions[i])
                ground_truth.append(targets[i])
                imgs[k, :, :, :] = images[i]
                k += 1

        iters += 1

    plot_predictions(imgs, preds, ground_truth, classes, 5, 3)


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
    
class TransfDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)