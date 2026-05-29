import pandas as pd
import numpy as np
import torch
import os
import sys

OUTPUT_DIR = './output'
feat_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'feature_matrix.csv'))

ESM_EMBEDDING_PATH = os.path.join(OUTPUT_DIR, 'esm2_embeddings.npy')

if os.path.exists(ESM_EMBEDDING_PATH):
    embeddings = np.load(ESM_EMBEDDING_PATH)
    print(f'Embeddings already exist: {embeddings.shape}')
    sys.exit(0)

print('Loading ESM-2 model...')
import esm

model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f'Using device: {device}')
model = model.to(device)

sequences = feat_df['vh_sequence'].values
n_seqs = len(sequences)
embed_dim = 480
embeddings = np.zeros((n_seqs, embed_dim), dtype=np.float32)

batch_size = 4
print(f'Encoding {n_seqs} sequences in batches of {batch_size}...')

for start in range(0, n_seqs, batch_size):
    end = min(start + batch_size, n_seqs)
    batch_seqs = [(f'seq_{i}', str(sequences[i])) for i in range(start, end)]

    try:
        with torch.no_grad():
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_seqs)
            batch_tokens = batch_tokens.to(device)
            results = model(batch_tokens, repr_layers=[12])
            token_representations = results['representations'][12]

        for i, (label, seq) in enumerate(batch_seqs):
            seq_len = len(seq)
            embeddings[start + i] = token_representations[i, 1:seq_len + 1].mean(dim=0).cpu().numpy()
    except Exception as e:
        print(f'Error at batch {start}-{end}: {e}')
        for i in range(start, end):
            seq = str(sequences[i])
            try:
                with torch.no_grad():
                    data = [(f'seq_{i}', seq)]
                    _, _, tokens = batch_converter(data)
                    tokens = tokens.to(device)
                    result = model(tokens, repr_layers=[12])
                    repr = result['representations'][12]
                    embeddings[i] = repr[0, 1:len(seq) + 1].mean(dim=0).cpu().numpy()
            except Exception as e2:
                print(f'  Failed seq {i}: {e2}')
                embeddings[i] = np.zeros(embed_dim, dtype=np.float32)

    if (start // batch_size) % 50 == 0:
        print(f'  Processed {end}/{n_seqs} sequences ({end/n_seqs*100:.1f}%)')

np.save(ESM_EMBEDDING_PATH, embeddings)
print(f'Done! Embeddings saved: {embeddings.shape}')
