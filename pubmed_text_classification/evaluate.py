import json
import os
from time import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix


def _get_scores(y_test, y_pred_test):
    scores = {}
    y_pred_test = np.concatenate(y_pred_test)
    scores['accuracy'] = accuracy_score(y_test, y_pred_test)
    scores['confusion_matrix'] = confusion_matrix(y_test, y_pred_test).tolist()
    return scores


def _save_results(save_dir, scores, pretrained_weights, num_train, valid_split, num_test, batch_size, num_epochs, lr,
                  optimizer, criterion, train_embeddings, **model_params):
    results = scores
    results['weights'] = pretrained_weights.split('/')[-1]
    results['num_train'] = num_train
    results['valid_split'] = valid_split
    results['num_test'] = num_test
    results['batch_size'] = batch_size
    results['num_epochs'] = num_epochs
    results['optimiser'] = str(optimizer)
    results['learning_rate'] = lr
    results['loss_fn'] = str(criterion)
    results['train_embeddings'] = train_embeddings
    results['model_params'] = {**model_params}
    timestamp = '{:.0f}'.format(time())

    model_saves_dir = os.path.join(save_dir, 'model_saves')
    results_save_dir = os.path.join(save_dir, 'results')
    for directory in [save_dir, results_save_dir, model_saves_dir]:
        if not os.path.exists(directory):
            os.mkdir(directory)

    # shutil.copyfile(VAL_SAVEPATH, os.path.join(model_saves_dir, '{}.pickle'.format(timestamp)))
    with open(os.path.join(results_save_dir, '{}.json'.format(timestamp)), 'w+') as outfile:
        json.dump(results, outfile)


def _predict(model, testloader):
    preds = []
    with torch.no_grad():
        model.eval()
        for X_i, y_i in testloader:
            y_hat_i = model(X_i)
            _, predicted = torch.max(y_hat_i.data, 1)
            predicted = predicted.cpu()
            preds.append(predicted.numpy())
            del X_i, y_hat_i, predicted
            torch.cuda.empty_cache()
    return preds