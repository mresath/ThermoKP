"""
===========================================================================
UniKP Inference Wrapper
Phase: Benchmarking Cross-Model Comparison
Description: Executes UniKP predictions against the ThermoKP benchmark dataset.
===========================================================================

Workflow:
1. Connect to the ThermoKP database and retrieve benchmark sequences.
2. Initialize UniKP's models (kcat and Km) and load pre-trained weights.
3. Process mature sequences and canonical SMILES using ProtT5 and SMILES Transformer.
4. Export predicted values to the central external_predictions directory.

Known Caveats:
- UniKP predicts log10(value), which is converted to natural log for standard evaluation.
- SMILES sequences longer than 218 tokens are automatically truncated per UniKP defaults.
- Utilizes an isolated Python 3.9 venv due to scikit-learn==0.24.2 compatibility issues.

Author: ThermoKP Team
License: MIT
"""

import os
import sys
import math
import sqlite3
import logging
import pandas as pd
import numpy as np
import torch
import gc
import re
import pickle
from pathlib import Path
from transformers.models.t5.modeling_t5 import T5EncoderModel  # type: ignore
from transformers.models.t5.tokenization_t5 import T5Tokenizer  # type: ignore
from rdkit import Chem

# ═══════════════════════════════════════════════════════════════════════════
#  Logging Configuration
# ═══════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  Path & Environment Setup
# ═══════════════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNIKP_DIR = PROJECT_ROOT / "external" / "unikp" / "src"
MODELS_DIR = PROJECT_ROOT / "external" / "unikp" / "UniKP"

import sys
_original_cwd = os.getcwd()
os.chdir(UNIKP_DIR)
sys.path.insert(0, str(UNIKP_DIR))
from build_vocab import WordVocab  # type: ignore
from pretrain_trfm import TrfmSeq2seq  # type: ignore
from utils import split  # type: ignore
sys.path.pop(0)
os.chdir(_original_cwd)

# ═══════════════════════════════════════════════════════════════════════════
#  Vectorization Methods
# ═══════════════════════════════════════════════════════════════════════════
def smiles_to_vec(Smiles):
    pad_index = 0
    unk_index = 1
    eos_index = 2
    sos_index = 3
    
    os.chdir(UNIKP_DIR)
    vocab = WordVocab.load_vocab('vocab.pkl')
    os.chdir(_original_cwd)
    
    def get_inputs(sm):
        seq_len = 220
        sm = sm.split()
        if len(sm) > 218:
            sm = sm[:109] + sm[-109:]
        ids = [vocab.stoi.get(token, unk_index) for token in sm]
        ids = [sos_index] + ids + [eos_index]
        seg = [1] * len(ids)
        padding = [pad_index] * (seq_len - len(ids))
        ids.extend(padding)
        seg.extend(padding)
        return ids, seg
        
    def get_array(smiles):
        x_id, x_seg = [], []
        for sm in smiles:
            a, b = get_inputs(sm)
            x_id.append(a)
            x_seg.append(b)
        return torch.tensor(x_id), torch.tensor(x_seg)
        
    trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    trfm.load_state_dict(torch.load(str(UNIKP_DIR / 'trfm_12_23000.pkl'), weights_only=True))
    trfm.eval()
    
    x_split = [split(sm) for sm in Smiles]
    xid, _ = get_array(x_split)
    X = trfm.encode(torch.t(xid))
    if isinstance(X, np.ndarray):
        return X
    return X.cpu().detach().numpy()

def Seq_to_vec(Sequence):
    for i in range(len(Sequence)):
        if len(Sequence[i]) > 1000:
            Sequence[i] = Sequence[i][:500] + Sequence[i][-500:]
            
    sequences_Example = []
    for i in range(len(Sequence)):
        zj = ' '.join(Sequence[i])
        sequences_Example.append(zj)
        
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)  # type: ignore
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")  # type: ignore
    gc.collect()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)  # type: ignore
    model = model.eval()
    
    features = []
    for i in range(len(sequences_Example)):
        sequences_Example_i = sequences_Example[i:i+1]
        sequences_Example_i = [re.sub(r"[UZOB]", "X", s) for s in sequences_Example_i]
        
        ids = tokenizer.batch_encode_plus(sequences_Example_i, add_special_tokens=True, padding=True)  # type: ignore
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)  # type: ignore
            
        embedding = embedding.last_hidden_state.cpu().numpy()
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum().item()
            seq_emd = embedding[seq_num][:seq_len - 1]
            features.append(seq_emd)
            
    features_normalize = np.zeros([len(features), len(features[0][0])], dtype=float)
    for i in range(len(features)):
        for k in range(len(features[0][0])):
            for j in range(len(features[i])):
                features_normalize[i][k] += features[i][j][k]
            features_normalize[i][k] /= len(features[i])
            
    return features_normalize

# ═══════════════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    logger.info("===========================================================================")
    logger.info("Starting UniKP Inference")
    logger.info("===========================================================================")

    with open(MODELS_DIR / 'UniKP for kcat.pkl', "rb") as f:
        kcat_model = pickle.load(f)
    with open(MODELS_DIR / 'UniKP for Km.pkl', "rb") as f:
        km_model = pickle.load(f)

    inputs_path = PROJECT_ROOT / "data" / "external_predictions" / "benchmark_inputs.csv"
    if not inputs_path.exists():
        logger.error(f"Input file not found at {inputs_path}. Please run export_benchmark_csv.py first.")
        return
        
    df = pd.read_csv(inputs_path)

    results = []
    batch_size = 32
    total_batches = math.ceil(len(df) / batch_size)
    
    for i in range(0, len(df), batch_size):
        batch_num = i // batch_size + 1
        logger.info(f"[{batch_num}/{total_batches}] Processing batch")
        
        batch_df = df.iloc[i:i+batch_size]
        
        valid_entries = []
        valid_seqs = []
        valid_smiles = []
        
        for _, row in batch_df.iterrows():
            entry_id = row["entry_id"]
            seq = row["sequence"]
            smiles = row["smiles"]
            
            # pandas treats missing strings as float NaN
            if isinstance(seq, str) and isinstance(smiles, str) and smiles != 'None' and "." not in smiles:
                valid_entries.append(entry_id)
                valid_seqs.append(seq)
                valid_smiles.append(smiles)
            else:
                results.append({
                    "entry_id": entry_id,
                    "log_kcat_pred": np.nan,
                    "log_km_pred": np.nan,
                    "kcat_pred": np.nan,
                    "km_pred": np.nan
                })
                
        if not valid_entries:
            continue
            
        seq_vec = Seq_to_vec(valid_seqs)
        smiles_vec = smiles_to_vec(valid_smiles)
        fused_vector = np.concatenate((smiles_vec, seq_vec), axis=1)
        
        kcat_pre_label = kcat_model.predict(fused_vector)
        km_pre_label = km_model.predict(fused_vector)
        
        for j, entry_id in enumerate(valid_entries):
            log10_kcat = kcat_pre_label[j]
            kcat_val = math.pow(10, log10_kcat)
            log_kcat_val = math.log(kcat_val) 
            
            log10_km = km_pre_label[j]
            km_val = math.pow(10, log10_km)
            log_km_val = math.log(km_val) 
            
            results.append({
                "entry_id": entry_id,
                "log_kcat_pred": log_kcat_val,
                "log_km_pred": log_km_val,
                "kcat_pred": kcat_val,
                "km_pred": km_val
            })
            
    out_dir = PROJECT_ROOT / "data" / "external_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "unikp_benchmark_predictions.csv"
    
    res_df = pd.DataFrame(results)
    res_df = df[['entry_id']].merge(res_df, on='entry_id', how='left')
    res_df.to_csv(out_path, index=False)
    
    logger.info("===========================================================================")
    logger.info(f"Inference complete. {len(df)} rows processed.")
    logger.info(f"Results exported to {out_path}")
    logger.info("===========================================================================")

if __name__ == "__main__":
    main()
