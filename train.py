import numpy as np
import pandas as pd
import json
import argparse
import os
import thinc
import random
import spacy
import sys
import GPUtil
import torch
from spacy.util import minibatch
from tqdm import tqdm
import unicodedata
import wasabi
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sagemaker_containers.beta.framework import (
    content_types,
    encoders,
    env,
    modules,
    transformer,
    worker,
)

def evaluate(nlp, texts, cats, batch_size):
    tp = 0.0  # True positives
    fp = 0.0  # False positives
    fn = 0.0  # False negatives
    tn = 0.0  # True negatives
    total_words = sum(len(text.split()) for text in texts)
    with tqdm(total=total_words, leave=False) as pbar:
        for i, doc in enumerate(nlp.pipe(texts, batch_size=batch_size)):
            gold = cats[i]
            for label, score in doc.cats.items():
                if label not in gold:
                    continue
                if score >= 0.5 and gold[label] >= 0.5:
                    tp += 1.0
                elif score >= 0.5 and gold[label] < 0.5:
                    fp += 1.0
                elif score < 0.5 and gold[label] < 0.5:
                    tn += 1
                elif score < 0.5 and gold[label] >= 0.5:
                    fn += 1
            pbar.update(len(doc.text.split()))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    if (precision + recall) == 0:
        f_score = 0.0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    return {"textcat_p": precision, "textcat_r": recall, "textcat_f": f_score}

# https://towardsdatascience.com/adaptive-and-cyclical-learning-rates-using-pytorch-2bf904d18dee
def cyclic_triangular_rate(min_lr, max_lr, period):
    it = 1
    while True:
        cycle = np.floor(1 + it / (2 * period))
        x = np.abs(it / period - 2 * cycle + 1)
        relative = max(0, 1 - x)
        yield min_lr + (max_lr - min_lr) * relative
        it += 1

TRAIN_DATA_FRACTION = 0.001
TEST_DATA_FRACTION = 0.001
TRAIN_SIZE = 0.6
LABEL_COL = "genre"
TEXT_COL = "sentence1"

def transform_data(data, train_size=0.6, train_frac=0.001, test_frac=0.001):
    """Extract the pandas data frame to a type that the model expects."""
    df_train, df_test = train_test_split(data, train_size = train_size, random_state=0)
    df_train = df_train.sample(frac=train_frac).reset_index(drop=True)
    df_test = df_test.sample(frac=test_frac).reset_index(drop=True)
    # Encode labels - maybe remove V
    labels = list(data[data.columns.difference([TEXT_COL])].columns)
    label_encoder = LabelEncoder()
    labels_train = label_encoder.fit_transform(labels)
    labels_test = label_encoder.transform(labels)
    
    num_labels = len(np.unique(labels_train))
    print("Number of unique labels: {}".format(num_labels))
    print("Number of training examples: {}".format(df_train.shape[0]))
    print("Number of testing examples: {}".format(df_test.shape[0]))
    eval_texts = df_test[TEXT_COL]
    eval_cats = json.loads(df_test[df_test.columns.difference([TEXT_COL])].to_json(orient='records'))
    train_texts = df_train[TEXT_COL]
    train_cats = json.loads(df_train[df_train.columns.difference([TEXT_COL])].to_json(orient='records'))
    train_data = list(zip(df_train[TEXT_COL].values, [{"cats": cats} for cats in train_cats]))
    
    return (train_data), (train_texts, train_cats), (eval_texts, eval_cats)

def train(data, model_dir, args):
    spacy.util.fix_random_seed(0)
    is_using_gpu = spacy.prefer_gpu()
    if is_using_gpu:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        print("GPU Usage")
        GPUtil.showUtilization()
    (train_data), (train_texts, train_cats), (eval_texts, eval_cats) = transform_data(data) # Update with argsparse
    
    model_choice = "en_pytt_xlnetbasecased_lg" #@param ["en_pytt_bertbaseuncased_lg", "en_pytt_xlnetbasecased_lg"]
    nlp = spacy.load(model_choice)
    print(f"Loaded model '{model_choice}'")
    if model_choice == "en_pytt_xlnetbasecased_lg":
      textcat = nlp.create_pipe(
              "pytt_textcat", config={"architecture": "softmax_class_vector"}
          )
    elif model_choice == "en_pytt_bertbaseuncased_lg":
      textcat = nlp.create_pipe(
              "pytt_textcat", config={"architecture": "softmax_class_vector"}
          )
    else: 
      raise Exception("Choose a supported PyTT model")
    labels = data[data.columns.difference([TEXT_COL])].columns
    for label in labels:
        print("Adding {}".format(label))
        textcat.add_label(label)
        
    print("Labels:", textcat.labels)
    nlp.add_pipe(textcat, last=True)
    print(f"Using {len(train_texts)} training docs, {len(eval_texts)} evaluation")
    
    # Initialize the TextCategorizer, and create an optimizer.
    optimizer = nlp.resume_training()
    optimizer.alpha = 0.001
    optimizer.pytt_weight_decay = 0.005
    optimizer.L2 = 0.0
    results = []
    epoch = 0
    step = 0
    eval_every = args.eval_every
    patience = args.patience
    batch_size = args.batch_size
    
    learn_rates = cyclic_triangular_rate(
        args.learn_rate / 3, args.learn_rate * 3, 2 * len(train_data) // batch_size
        )
    print("Training the model...")
    print("{:^5}\t{:^5}\t{:^5}\t{:^5}".format("LOSS", "P", "R", "F"))

    # pbar = tqdm(total=100, leave=False)

    msg = wasabi.Printer()
    with tqdm(total=100, leave=False, file=sys.stdout) as pbar:
        while True:
            # Train and evaluate
            losses = Counter()
            random.shuffle(train_data)
            batches = minibatch(train_data, size=batch_size)
            for batch in batches:
        #         print('Batch {}'.format(batch))
                optimizer.pytt_lr = next(learn_rates)
                texts, annotations = zip(*batch)
        #         print('Updating {}'.format(texts))
                nlp.update(texts, annotations, sgd=optimizer, drop=0.1, losses=losses)
                pbar.update(1)
    #             msg.info(f"Step - {step} : Step % Eval Every - {(step % eval_every)}")
                if step and (step % eval_every) == 0:
                    pbar.close()
                    with nlp.use_params(optimizer.averages):
                        scores = evaluate(nlp, eval_texts, eval_cats, batch_size)
                    results.append((scores["textcat_f"], step, epoch))
                    msg.warn(
                        "Losses: [{0:.3f}] Precision: [{1:.3f}] Recall: [{2:.3f}] F1-Score: [{3:.3f}]".format(
                            losses["pytt_textcat"],
                            scores["textcat_p"],
                            scores["textcat_r"],
                            scores["textcat_f"],
                        )
                    )
                    pbar = tqdm(total=eval_every, leave=False)
                step += 1
            epoch += 1
            msg.good(f"=== Epoch {epoch} ===")
            # Stop if no improvement in HP.patience checkpoints
            if results:
                best_score, best_step, best_epoch = max(results)
                msg.info(f"best score: {best_score}  best_step : {best_step}  best epoch : {best_epoch} ")
                msg.info(f"break clause: {((step - best_step) // eval_every)}")
                msg.good(f"(Step - Best step: [{step - best_step}] Eval Every: [{eval_every}] Patience: [{patience}]")
                if ((step - best_step) // eval_every) >= patience:
                    break

            table_widths = [2, 4, 6]
            msg.info(f"Best scoring checkpoints")
            msg.row(["Epoch", "Step", "Score"], widths=table_widths)
            msg.row(["-" * width for width in table_widths])
            for score, step, epoch in sorted(results, reverse=True)[:10]:
                msg.row([epoch, step, "%.2f" % (score * 100)], widths=table_widths)

            # Test the trained model
            test_text = eval_texts[0]
            doc = nlp(test_text)
            print(test_text, doc.cats)
    nlp.to_disk(model_dir)
    print("Finished training. Model saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--eval-every', type=int, default=3, metavar='EE',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--learn-rate', type=float, default=2e-5, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--patience', type=float, default=3, metavar='P',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--model-name', type=str, default='en_pytt_xlnetbasecased_lg',
                        help='model to use (en_pytt_xlnetbasecased_lg, en_pytt_bertbaseuncased_lg, en_pytt_xlnetbasecased_lg)')

    # Container environment
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    args = parser.parse_args()

    # Take the set of files and read them all into a single pandas dataframe
    input_files = [
        os.path.join(args.data_dir, file)
        for file in os.listdir(args.data_dir)
        if file.endswith("json")
    ]
    print(input_files)
    if len(input_files) == 0:
        raise ValueError(
            (
                "There are no files in {}.\n"
                + "This usually indicates that the channel ({}) was incorrectly specified,\n"
                + "the data specification in S3 was incorrectly specified or the role specified\n"
                + "does not have permission to access the data."
            ).format(args.train, "train")
        )

    raw_data = [pd.read_json(file) for file in input_files]
    print("Number of data frames length: ", len(raw_data))
    concat_data = pd.concat(raw_data)

    print("Concat data")

    print(concat_data.head(2))

    print("Length of data ", len(concat_data))

    print("Frame columns", concat_data.columns)

    train(concat_data, os.path.join(args.model_dir, "model.joblib"), args)


def model_fn(model_dir):
    """Deserialize fitted model
    """
    nlp = spacy.load("en_pytt_xlnetbasecased_lg")
    nlp.from_disk(os.path.join(model_dir, "model.joblib"))
#     model = joblib.load(os.path.join(model_dir, "model.joblib"))
#     print("[model_fn] - Celebrities data", model["celebrities"].head(3))
    print("[model_fn] ",nlp)
    return nlp


def input_fn(input_data, content_type):
    """Takes request data and de-serializes the data into an object for prediction.
        When an InvokeEndpoint operation is made against an Endpoint running SageMaker model server,
        the model server receives two pieces of information:
            - The request Content-Type, for example "application/json"
            - The request data, which is at most 5 MB (5 * 1024 * 1024 bytes) in size.
        The input_fn is responsible to take the request data and pre-process it before prediction.
    Args:
        input_data (obj): the request data.
        content_type (str): the request Content-Type.
    Returns:
        (obj): data ready for prediction.
    """
    print("[input_fn] - Input type:", type(input_data))
    if content_type == "application/json":
        return json.loads(input_data)
    else:
        raise ValueError(
            "Not yet supported by script! Current support is only content-type: [application/json]"
        )


def predict_fn(input_data, model):
    """Preprocess input data
    
    We implement this because the default predict_fn uses .predict(), but our model is a preprocessor
    so we want to use .transform().

    The output is returned in the following order:
    
        rest of features either one hot encoded or standardized
    """
    print("[predict_fn] Input Type: ", type(input_data))
    print("[predict_fn] Input: ", input_data)
    print("[predict_fn] Model: ", model)
    nlp = model(input_data)
    return nlp.cats


def output_fn(prediction, accept):
    """Format prediction output
    
    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    """
    print("[output_fn] Prediction: ", prediction)
    print("[output_fn] Accept: ", accept)
    return worker.Response(encoders.encode(prediction, accept), accept, mimetype=accept)