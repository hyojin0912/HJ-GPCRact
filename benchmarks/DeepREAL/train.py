"""
DeepFunLib based on DISAE (DeepREAL Benchmark)
Protein descriptor: DISAE-plus
Chemical descriptor: contextPred
Data: GLASS (Agonist/Antagonist/Non-binder)
"""

import warnings
warnings.filterwarnings('ignore')
import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef as mcc
from rdkit import Chem

# Local imports (Ensure these files exist in the same directory)
from data_tool_box import *
from models import DeepREALModel, DTI_model_pretrained  # Renamed class import
from utils import *

# -------------------------------
# Early Stopping Class
# -------------------------------
class EarlyStopping:
    """Stop training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, delta=0, output_dir='./', filename='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.output_dir = output_dir
        self.filename = filename

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), os.path.join(self.output_dir, self.filename))
        self.val_loss_min = val_loss

def core_batch_prediction(traindf, i, all_config, tokenizer, chem_dict, protein_dict, model, epoch, by_epoch=False, detach=True):
    # ----------------------------------
    #           Process Input
    # ----------------------------------
    if by_epoch:
        batch_data = traindf[i * all_config['batch_size']:(i + 1) * all_config['batch_size']]
    else:
        batch_data = traindf.sample(all_config['batch_size'])

    batch_data, batch_chem_graphs, batch_protein_tokenized = get_repr_DTI(
        batch_data, tokenizer, chem_dict, protein_dict,
        all_config['prot_descriptor'], 'contextpred'
    )

    if all_config['use_cuda'] and torch.cuda.is_available():
        batch_protein_tokenized = batch_protein_tokenized.to('cuda')
        batch_chem_graphs = batch_chem_graphs.to('cuda')

    # ----------------------------------
    #       Get Prediction Score
    # ----------------------------------
    batch_logits = model(batch_protein_tokenized, batch_chem_graphs, epoch)

    # ----------------------------------
    #            Loss
    # ----------------------------------
    batch_labels = torch.LongTensor(batch_data['Activity'].values)
    if all_config['use_cuda'] and torch.cuda.is_available():
        batch_labels = batch_labels.to('cuda')
        
    if detach:
        batch_logits = batch_logits.detach().cpu().numpy()
        batch_labels = batch_labels.detach().cpu().numpy()

    return batch_data, batch_logits, batch_labels

def evaluate_3class(df, all_config, tokenizer, chem_dict, protein_dict, model, datatype='dev'):
    output = pd.DataFrame()
    collected_labels = np.empty((0))
    collected_logits = np.empty((0, 3))
    epoch = 1000 # Dummy epoch for evaluation mode

    for i in range(int(df.shape[0] / all_config['batch_size'])):
        batch_data, batch_logits, batch_labels = core_batch_prediction(
            df, i, all_config, tokenizer, chem_dict, protein_dict, model,
            epoch, by_epoch=True, detach=True)
        
        output = pd.concat([output, batch_data], axis=0)
        collected_labels = np.append(collected_labels, batch_labels, axis=0)
        collected_logits = np.append(collected_logits, batch_logits, axis=0)

    # Calculate Probabilities
    logits_tensor = torch.tensor(collected_logits, dtype=torch.float32)
    probabilities = F.softmax(logits_tensor, dim=1).numpy()

    prob_df = pd.DataFrame(probabilities, columns=['prob0', 'prob1', 'prob2'])
    output = pd.concat([output.reset_index(drop=True), prob_df], axis=1)
    return output

class Trainer_3class():
    def __init__(self, binary_model=None, tokenizer=None, all_config=None, checkpoint_dir=None):
        # ----------------------------------
        #    Hyper-parameter / Config
        # ----------------------------------
        self.checkpoint_dir = checkpoint_dir
        self.opt_config = all_config['opt_config']
        self.admin_config = all_config['admin_config']
        self.all_config = all_config
        
        # ----------------------------------
        #       Model
        # ----------------------------------
        self.model = DeepREALModel(all_config=all_config, DTI_binary_pretrained=binary_model)
        if self.all_config['use_cuda'] and torch.cuda.is_available():
            self.model = self.model.to('cuda')
        self.tokenizer = tokenizer
        
        # ----------------------------------
        #       Input Data Loading
        # ----------------------------------
        chem_path = os.path.join(all_config['cwd'], 'data/chemical/ikey2smiles.csv')
        df_chem = pd.read_csv(chem_path)
        df_chem = df_chem.drop_duplicates(subset='ikey')
        self.chem_dict = dict(zip(df_chem['ikey'], df_chem['smiles']))
        
        prot_path = os.path.join(all_config['cwd'], 'data/protein/Uniprot2Triplet_Combined_Final_split_253.csv')
        self.uni2triplet = pd.read_csv(prot_path)
        self.protein_dict = dict(zip(self.uni2triplet.UNIPROT_AC, self.uni2triplet.Triplet))
        
        print('Trainer initialized.')

    def train(self):
        # ----------------------------------
        #    Load Interaction Data
        # ----------------------------------
        data_path = os.path.join(self.all_config['cwd'], 'data/interaction/the_3way_chem_split/')
        traindf, devdf_eval, testdf_eval = load_training_data(data_path)

        traindf = filter_invalid_pairs(traindf, self.chem_dict, self.protein_dict)
        devdf_eval = filter_invalid_pairs(devdf_eval, self.chem_dict, self.protein_dict)
        testdf_eval = filter_invalid_pairs(testdf_eval, self.chem_dict, self.protein_dict)

        # ----------------------------------
        #    Training Setup
        # ----------------------------------
        parameters = list(self.model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=self.all_config['lr'], weight_decay=self.opt_config['l2'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        loss_fn = torch.nn.CrossEntropyLoss()
        
        # ----------------------------------
        #           Training Loop
        # ----------------------------------
        early_stopping = EarlyStopping(patience=10, verbose=True, output_dir=self.checkpoint_dir)
        stime = time.time()
        
        for epoch in range(self.all_config['epochs']):
            print(f'------------------------ Epoch: {epoch} ------------------------')
            
            # Training Step
            self.model.train()
            for i in range(int(traindf.shape[0] / self.all_config['batch_size'])):
                _, batch_logits, batch_labels = core_batch_prediction(
                    traindf, i, self.all_config, self.tokenizer,
                    self.chem_dict, self.protein_dict, self.model, epoch,
                    detach=False, by_epoch=True
                )
                loss = loss_fn(batch_logits, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            
            # Validation Step (Early Stopping)
            loss_list_dev = []
            self.model.eval()
            for i in range(int(devdf_eval.shape[0] / self.all_config['batch_size'])):
                _, batch_logits, batch_labels = core_batch_prediction(
                    devdf_eval, i, self.all_config, self.tokenizer,
                    self.chem_dict, self.protein_dict, self.model, epoch,
                    detach=False, by_epoch=True
                )
                loss = loss_fn(batch_logits, batch_labels)
                loss_list_dev.append(loss.item())
            
            valid_loss = np.average(loss_list_dev)
            early_stopping(valid_loss, self.model)

            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

            # ----------------------------------
            #           Evaluation
            # ----------------------------------
            self.model.eval()
            traindf_eval = traindf.sample(frac=1) 
            
            print(f'Eval Size -> Train: {traindf_eval.shape[0]}, Dev: {devdf_eval.shape[0]}, Test: {testdf_eval.shape[0]}')
            
            output_train = evaluate_3class(traindf_eval, self.all_config, self.tokenizer, self.chem_dict, self.protein_dict, self.model, datatype='train')
            output_dev = evaluate_3class(devdf_eval, self.all_config, self.tokenizer, self.chem_dict, self.protein_dict, self.model, datatype='dev')
            output_test = evaluate_3class(testdf_eval, self.all_config, self.tokenizer, self.chem_dict, self.protein_dict, self.model, datatype='test')

            # Save Predictions
            output_train.to_csv(os.path.join(self.checkpoint_dir, f'epoch{epoch}_train_output.csv'), index=None)
            output_dev.to_csv(os.path.join(self.checkpoint_dir, f'epoch{epoch}_dev_output.csv'), index=None)
            output_test.to_csv(os.path.join(self.checkpoint_dir, f'epoch{epoch}_test_output.csv'), index=None)

            print(f'Time cost of the epoch: {time.time() - stime:.2f}s')
            stime = time.time()

# -------------------------------------------
#      Helper Function
# -------------------------------------------
def filter_invalid_pairs(df, chem_dict, protein_dict):
    invalid_rows = []
    for idx, row in df.iterrows():
        ikey = row['ikey']
        uni = row['uniprot']
        smiles = chem_dict.get(ikey, None)

        if not isinstance(smiles, str):
            invalid_rows.append(idx)
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid_rows.append(idx)
            continue

        if uni not in protein_dict:
            invalid_rows.append(idx)

    if invalid_rows:
        print(f"Removing {len(invalid_rows)} invalid rows...")
        df = df.drop(index=invalid_rows).reset_index(drop=True)
    return df

# -------------------------------------------
#      Main Execution
# -------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser("DeepREAL Benchmark")
    parser.add_argument('--cwd', type=str, default='./', help='Current Working Directory')
    parser.add_argument('--debug_ratio', type=float, default=1.0)
    parser.add_argument('--exp_mode', default='the_3way_chem_split/', help='Path to the train/dev/test dataset.')
    parser.add_argument('--pretrained_binary_path', default='binary_pretrain/')
    
    # Protein Descriptor Args
    parser.add_argument('--prot_descriptor', type=str, default='DISAE', help='choose from [DISAE, TAPE, ESM]')
    parser.add_argument('--DISAE_raw', type=str2bool, default=False)
    parser.add_argument('--prot_frozen', type=str, default='none', help='choose from {whole, none, partial}')
    parser.add_argument('--frozen', type=str, default='none', help='choose from {whole, none, partial}')
    parser.add_argument('--binary_frozen', type=str, default='none-none-none', help='{none, whole-whole-whole}')
    parser.add_argument('--pretrained_onBinary', type=str2bool, nargs='?', const=True, default=True)
    
    # Training Args
    parser.add_argument('--epochs', default=100, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=64, type=int, help="Batch size")
    parser.add_argument('--use_cuda', type=str2bool, nargs='?', const=True, default=True, help='use cuda.')
    parser.add_argument('--lr', type=float, default=2e-5, help="Initial learning rate")

    opt = parser.parse_args()

    # Load Config
    config_path = os.path.join(opt.cwd, 'DTI_config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    all_config = load_json(config_path)
    checkpoint_dir = set_up_exp_folder(opt.cwd, 'results/')
    
    np.random.seed(7)
    seed = all_config['opt_config']['random_seed']
    torch.manual_seed(seed)
    if opt.use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    all_config.update(vars(opt))
    save_json(all_config, os.path.join(checkpoint_dir, 'config.json'))

    if not opt.use_cuda:
        print('Warning: Not using GPU')

    # Load Protein Descriptor (DISAE)
    if all_config['prot_descriptor'] == 'DISAE':
        print('Using DISAE+ descriptor')
        from transformers import AlbertConfig, BertTokenizer, AlbertForMaskedLM
        from transformers import load_tf_weights_in_albert 

        albert_config_path = os.path.join(all_config['cwd'], all_config['DISAE']['albertconfig'])
        albertconfig = AlbertConfig.from_pretrained(albert_config_path)
        m = AlbertForMaskedLM(config=albertconfig)
        
        if not all_config['DISAE_raw']:
            # Load Pretrained TF weights logic if needed, or PyTorch weights
            # Simplified for benchmark: assuming weights are loadable or skipping raw load
            pass 
        
        prot_descriptor = m.albert
        vocab_path = os.path.join(all_config['cwd'], all_config['DISAE']['albertvocab'])
        prot_tokenizer = BertTokenizer.from_pretrained(vocab_path)

    # Initialize Model
    print("Initializing DeepREAL Model...")
    DTI_model_pretrained_instance = DTI_model_pretrained(all_config=all_config, model=prot_descriptor)
    
    if all_config['pretrained_onBinary']:
        print('Loading pretrained binary model...')
        # Adjusted path logic for benchmark folder structure
        # Assumes the pretrained model file is placed in 'models/pretrained/model.dat'
        binary_ckpt = os.path.join(all_config['cwd'], 'models', 'pretrained', 'binary_model.dat')
        if os.path.exists(binary_ckpt):
             DTI_model_pretrained_instance.load_state_dict(torch.load(binary_ckpt, map_location='cpu'))
        else:
             print(f"Warning: Pretrained binary model not found at {binary_ckpt}. initializing random weights.")

    # Start Training
    trainer = Trainer_3class(
        binary_model=DTI_model_pretrained_instance, 
        tokenizer=prot_tokenizer,
        all_config=all_config, 
        checkpoint_dir=checkpoint_dir
    )

    trainer.train()
    print(f'Finished training! Results saved to: {checkpoint_dir}')