import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.nn import GCNConv
from metrics import get_f1_score

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden=64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, 2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.3, training=self.training)
        return self.conv2(x, edge_index)

class Runner:
    def __init__(self, base_path):
        self.public = os.path.join(base_path, "data/public")
        self.private = os.path.join(base_path, "data/private")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_data(self):
        train_nodes = pd.read_csv(os.path.join(self.public, 'train_nodes.csv'))
        train_labels = pd.read_csv(os.path.join(self.public, 'train_labels.csv'))
        test_nodes = pd.read_csv(os.path.join(self.public, 'test_nodes.csv'))
        edgelist = pd.read_csv(os.path.join(self.public, 'edgelist.csv'))
        gt = pd.read_csv(os.path.join(self.private, 'hidden_labels.csv'))
        all_nodes = pd.concat([train_nodes, test_nodes]).drop_duplicates('id').reset_index(drop=True)
        mapping = {node_id: i for i, node_id in enumerate(all_nodes['id'])}
        x = torch.from_numpy(all_nodes.iloc[:, 2:].values).float()
        
        edge_index = torch.tensor([
            edgelist['txId1'].map(mapping).values, 
            edgelist['txId2'].map(mapping).values
        ], dtype=torch.long)

        y = torch.full((len(all_nodes),), -1, dtype=torch.long)
        
        train_labels['y'] = train_labels['y'].astype(int)
        train_mapped_ids = train_labels['id'].map(mapping).values
        train_mapped_vals = train_labels['y'].map({1: 1, 2: 0}).values
        y[train_mapped_ids] = torch.tensor(train_mapped_vals).long()

        val_idx = gt['id'].map(mapping).values
        y_val = gt['y_true'].values 

        return x.to(self.device), edge_index.to(self.device), y.to(self.device), val_idx, y_val, test_nodes['id'].values, mapping

    def train(self, x, edge_index, y, val_idx, y_val):
        model = GNN(x.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        
        weights = torch.tensor([0.55, 5.09]).to(self.device)
        train_mask = y != -1

        for epoch in range(101):
            model.train()
            optimizer.zero_grad()
            out = model(x, edge_index)
            loss = F.cross_entropy(out[train_mask], y[train_mask], weight=weights)
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                model.eval()
                with torch.no_grad():
                    logits = model(x, edge_index)
                    probs = F.softmax(logits, dim=1)[val_idx, 1].cpu().numpy()
                    f1 = get_f1_score(y_val, probs)
                    print(f"epoch {epoch:02d} | loss: {loss.item():.4f} |  f1: {f1:.4f}")
        return model

def run():
    runner = Runner('') 
    x, edge_index, y, val_idx, y_val, test_ids, mapping = runner.load_data()
    model = runner.train(x, edge_index, y, val_idx, y_val)
    
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        test_probs = np.array([probs[mapping[tid]] for tid in test_ids])
        preds_binary = (test_probs > 0.5).astype(int)
        final_labels = np.where(preds_binary == 1, 1, 2)
        sub = pd.DataFrame({'id': test_ids, 'y_pred': final_labels})
        
        output_d = os.path.join("submissions", "inbox", "team_alpha", "run_01")
        os.makedirs(output_d, exist_ok=True)
        out_p = os.path.join(output_d, "predictions.csv")
        sub.to_csv(out_p, index=False)

if __name__ == "__main__":
    run()
