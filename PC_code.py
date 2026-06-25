import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import to_dense_batch, add_self_loops, degree
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, 
    f1_score, balanced_accuracy_score, confusion_matrix,
    precision_score, recall_score, matthews_corrcoef,
    roc_curve, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from tqdm import tqdm
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh

DATA_DIR = "/kaggle/input/datasets/baby9966/pancreatic-cancer/"

warnings.filterwarnings('ignore')

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ChebConvII_Optimized(nn.Module):

    def __init__(self, in_channels, out_channels, K=5, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K

        self.cheb_weights = nn.Parameter(torch.Tensor(K, in_channels, out_channels))
        self.scale = nn.Parameter(torch.ones(1))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.cheb_weights)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_weight=None, lambda_max=None):
        num_nodes = x.size(0)

        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, fill_value=1.0, num_nodes=num_nodes
        )

        row, col = edge_index
        deg = degree(row, num_nodes, dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        adj_norm = torch.sparse_coo_tensor(
            edge_index, norm, size=(num_nodes, num_nodes)
        )

        def propagate(feat):
            return torch.sparse.mm(adj_norm, feat)

        if lambda_max is None:
            lambda_max = 2.0

        cheb_x = [x]

        if self.K > 1:
            Lx = x - propagate(x)
            L_tilde_x = 2.0 * Lx / lambda_max - x
            cheb_x.append(L_tilde_x)

        for k in range(2, self.K):
            Lx_prev = cheb_x[-1] - propagate(cheb_x[-1])
            L_tilde_x_prev = 2.0 * Lx_prev / lambda_max - cheb_x[-1]

            Lx_prev2 = cheb_x[-2] - propagate(cheb_x[-2])
            L_tilde_x_prev2 = 2.0 * Lx_prev2 / lambda_max - cheb_x[-2]

            cheb_x.append(2.0 * L_tilde_x_prev - L_tilde_x_prev2)

        out = torch.zeros(num_nodes, self.out_channels, device=x.device)
        for k in range(self.K):
            out += torch.mm(cheb_x[k], self.cheb_weights[k]) * self.scale

        if self.bias is not None:
            out += self.bias

        return out

class ChebNetII_Transformer_PESE(nn.Module):

    def __init__(self, in_dim=1, hidden_dim=128, out_dim=128, 
                 pe_k=8, se_walk=20, K=5, num_heads=4, 
                 num_transformer_layers=2, dropout=0.3):
        super().__init__()
        
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        self.pe_proj = nn.Linear(pe_k, hidden_dim)
        self.se_proj = nn.Linear(se_walk, hidden_dim)
        
        self.cheb_conv = ChebConvII_Optimized(hidden_dim, hidden_dim, K=K)
        
        self.transformer_layers = nn.ModuleList()
        for _ in range(num_transformer_layers):
            self.transformer_layers.append(
                TransformerConv(hidden_dim, hidden_dim, heads=num_heads, 
                               dropout=dropout, concat=False)
            )
        
        self.norm_cheb = nn.LayerNorm(hidden_dim)
        self.norm_trans = nn.LayerNorm(hidden_dim)
        self.norm_final = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.gene_importance = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.pool_query = nn.Parameter(torch.randn(1, hidden_dim))
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, edge_index, pe, se, edge_weight=None, lambda_max=None):
        h = self.input_proj(x)  # [N, D]
        
        pe_proj = self.pe_proj(pe) if pe is not None else None
        se_proj = self.se_proj(se) if se is not None else None
        
        if pe_proj is not None:
            h = h + pe_proj
        if se_proj is not None:
            h = h + se_proj
        
        h = self.dropout(h)
        
        h_cheb = self.cheb_conv(h, edge_index, edge_weight, lambda_max)
        h_cheb = F.silu(h_cheb)
        h_cheb = self.dropout(h_cheb)

        h = self.norm_cheb(h + h_cheb)
        
        for trans_layer in self.transformer_layers:
            h_trans = trans_layer(h, edge_index)
            h_trans = F.silu(h_trans)
            h_trans = self.dropout(h_trans)
            h = self.norm_trans(h + h_trans)
        
        importance = self.gene_importance(h).squeeze(-1)  
        
        attention_weights = F.softmax(importance, dim=0)  
        graph_repr = (h * attention_weights.unsqueeze(1)).sum(dim=0, keepdim=True)  
        
        logits = self.classifier(graph_repr)
        
        return logits, importance, h

def compute_laplacian_pe(edge_index, num_nodes, k=8):

    adj = np.zeros((num_nodes, num_nodes))
    edge_index_np = edge_index.cpu().numpy()
    for i in range(edge_index_np.shape[1]):
        u, v = edge_index_np[0, i], edge_index_np[1, i]
        adj[u, v] = 1.0
    adj = np.maximum(adj, adj.T)
    L = csgraph.laplacian(adj, normed=True)
    
    try:
        eigenvalues, eigenvectors = eigsh(L, k=min(k, num_nodes-1), which='SM')
        pe = torch.tensor(eigenvectors[:, :k], dtype=torch.float32)
    except:
        pe = torch.randn(num_nodes, k) * 0.01
    return pe


def compute_rwse(edge_index, num_nodes, walk_length=20):

    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    for i in range(edge_index.size(1)):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        adj[u, v] = 1.0
    
    deg = adj.sum(dim=1, keepdim=True)
    deg[deg == 0] = 1.0
    P = adj / deg
    
    rw_stats = []
    P_power = torch.eye(num_nodes)
    for _ in range(walk_length):
        P_power = P_power @ P
        rw_stats.append(P_power.diagonal())
    rwse = torch.stack(rw_stats, dim=1)
    return rwse


def compute_lambda_max(edge_index, num_nodes):

    adj = np.zeros((num_nodes, num_nodes))
    edge_index_np = edge_index.cpu().numpy()
    for i in range(edge_index_np.shape[1]):
        u, v = edge_index_np[0, i], edge_index_np[1, i]
        adj[u, v] = 1.0
    adj = np.maximum(adj, adj.T)
    L = csgraph.laplacian(adj, normed=True)
    
    try:
        lambda_max = eigsh(L, k=1, which='LM', return_eigenvectors=False)[0]
    except:
        lambda_max = 2.0
    return torch.tensor(lambda_max, dtype=torch.float32)


def load_sample_labels():

    df = pd.read_csv(DATA_DIR + "sample_info_GSE62452.csv")
    
    print(f"原始数据: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    
    if 'Dataset' in df.columns:
        df = df[df['Dataset'] == 'GSE62452'].copy()
        print(f"筛选后 (仅GSE54236): {df.shape}")
    
    print(f"\nGroup列唯一值:")
    print(df['Group'].value_counts())
    
    def get_label(group_value):
        group_str = str(group_value).lower().strip()
        if 'normal' in group_str:
            return 0
        elif 'tumor' in group_str:
            return 1
        else:
            print(f"警告: 无法识别的标签 '{group_value}'，默认为0")
            return 0
    
    df['label'] = df['Group'].apply(get_label)
    
    if 'Sample' in df.columns:
        df = df.rename(columns={'Sample': 'Sample_ID'})
    elif 'GSM' in df.columns:
        df = df.rename(columns={'GSM': 'Sample_ID'})
    
    df['Sample_ID'] = df['Sample_ID'].astype(str).str.strip()
    
    tumor_count = (df['label'] == 1).sum()
    normal_count = (df['label'] == 0).sum()
    
    print(f"\n样本分布:")
    print(f"  NonTumor: {normal_count}")
    print(f"  Tumor: {tumor_count}")
    print(f"  总计: {len(df)}")
    
    print(f"\n样本示例:")
    print(df[['Sample_ID', 'Group', 'label']].head(10))
    
    return df

def load_expression_data():
    expr = pd.read_csv(DATA_DIR + "GSE62452_expression.csv", index_col=0)
    print(f"\n表达矩阵: {expr.shape[0]} 基因 × {expr.shape[1]} 样本")
    
    expr.columns = expr.columns.astype(str).str.strip()
    print(f"前5个样本列名: {expr.columns[:5].tolist()}")
    
    return expr

def load_ppi_network():

    edges = pd.read_csv(DATA_DIR + "ppi_network_edges_threshold400.csv")
    nodes = pd.read_csv(DATA_DIR + "ppi_network_nodes_threshold400.csv")
    print(f"\nPPI网络:")
    print(f"  节点数: {len(nodes)}")
    print(f"  边数: {len(edges)}")
    print(f"  节点列名: {nodes.columns.tolist()}")
    print(f"  边列名: {edges.columns.tolist()}")
    return edges, nodes

def compute_edge_weight(edge_index_np, n_nodes):
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import eigsh
    edge_index_np = edge_index_np.astype(np.int64)
    adj = coo_matrix((np.ones(len(edge_index_np[0])), (edge_index_np[0], edge_index_np[1])), 
                     shape=(n_nodes, n_nodes))
    adj = adj + adj.T
    adj.data = np.ones(len(adj.data))
    degree = np.array(adj.sum(axis=1)).flatten()
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degree + 1e-8))
    L = np.eye(n_nodes) - D_inv_sqrt @ adj.toarray() @ D_inv_sqrt
    lambda_max = eigsh(L, k=1, which='LM', return_eigenvectors=False)[0]
    edge_weight = torch.ones(edge_index_np.shape[1], dtype=torch.float)
    return torch.tensor(L, dtype=torch.float), torch.tensor(lambda_max, dtype=torch.float), edge_weight


def prepare_data(sample_ids, expression_df, edges, nodes, label_dict, pe_k=8, se_walk=20):
    genes_in_network = nodes['gene'].tolist()
    n_nodes = len(genes_in_network)
    
    nodes = nodes.copy()
    nodes['index'] = nodes['index'].astype(np.int64)
    gene_to_idx = dict(zip(nodes['gene'], nodes['index']))
    
    if 'gene_A' not in edges.columns:
        if 'Gene1' in edges.columns:
            edges = edges.rename(columns={'Gene1': 'gene_A', 'Gene2': 'gene_B'})
        elif 'source' in edges.columns:
            edges = edges.rename(columns={'source': 'gene_A', 'target': 'gene_B'})
        elif 'gene1' in edges.columns:
            edges = edges.rename(columns={'gene1': 'gene_A', 'gene2': 'gene_B'})
    
    valid_edges = edges[edges['gene_A'].isin(genes_in_network) & edges['gene_B'].isin(genes_in_network)]
    print(f"有效边数: {len(valid_edges)}")
    
    src = np.array([gene_to_idx[row['gene_A']] for _, row in valid_edges.iterrows()], dtype=np.int64)
    dst = np.array([gene_to_idx[row['gene_B']] for _, row in valid_edges.iterrows()], dtype=np.int64)
    
    edge_index_np = np.stack([
        np.concatenate([src, dst]).astype(np.int64),
        np.concatenate([dst, src]).astype(np.int64)
    ], axis=0)
    edge_index_np = np.unique(edge_index_np, axis=1)
    edge_index = torch.from_numpy(edge_index_np).long()
    print(f"最终边数: {edge_index.shape[1]}")
    
    L, lambda_max, edge_weight = compute_edge_weight(edge_index_np, n_nodes)
    
    print("预计算 Laplacian PE ...")
    pe_raw = compute_laplacian_pe(edge_index, n_nodes, k=pe_k)
    print("预计算 Random Walk SE ...")
    se_raw = compute_rwse(edge_index, n_nodes, walk_length=se_walk)
    
    all_expr = []
    valid_sample_ids = []
    
    expression_columns = expression_df.columns.astype(str).str.strip().tolist()
    sample_ids_str = [str(sid).strip() for sid in sample_ids]
    
    print(f"\n样本匹配中...")
    print(f"表达矩阵样本数: {len(expression_columns)}")
    print(f"标签样本数: {len(sample_ids_str)}")
    
    print(f"标签样本ID示例: {sample_ids_str[:5]}")
    print(f"表达矩阵样本ID示例: {expression_columns[:5]}")
    
    for sample_id in sample_ids_str:
        if sample_id in expression_columns:
            expr_sample = expression_df[sample_id]
            x_values = expr_sample.reindex(genes_in_network).fillna(0).values.reshape(-1, 1)
            all_expr.append(x_values)
            valid_sample_ids.append(sample_id)
        else:
            print(f"警告: 样本 {sample_id} 不在表达矩阵中")
    
    if len(all_expr) == 0:
        raise ValueError("没有找到匹配的样本！请检查样本ID格式。")
    
    print(f"成功匹配 {len(all_expr)} 个样本")
    
    all_expr_concat = np.concatenate(all_expr, axis=0)
    scaler = StandardScaler()
    scaler.fit(all_expr_concat)
    
    data_list = []
    for i in tqdm(range(len(valid_sample_ids)), desc='Building graph data', leave=False):
        x_scaled = scaler.transform(all_expr[i])
        x = torch.tensor(x_scaled, dtype=torch.float)
        
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            pe=pe_raw,
            se=se_raw
        )
        data.L = L
        data.lambda_max = lambda_max
        data.y = torch.tensor([label_dict[valid_sample_ids[i]]], dtype=torch.long)
        data_list.append(data)
    
    return data_list, L, lambda_max, edge_index, edge_weight


def train_epoch(model, train_data_list, edge_index, edge_weight, labels, optimizer, criterion, device, scheduler=None):
    model.train()
    total_loss = 0
    
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device) if edge_weight is not None else None
    
    indices = np.random.permutation(len(train_data_list))
    
    for idx in indices:
        data = train_data_list[idx]
        label = labels[idx]
        
        x = data.x.to(device)
        pe = data.pe.to(device)
        se = data.se.to(device)
        lambda_max = data.lambda_max.to(device)
        y = torch.tensor([label], dtype=torch.long).to(device)
        
        optimizer.zero_grad()
        logits, _, _ = model(x, edge_index, pe, se, edge_weight, lambda_max)
        loss = criterion(logits, y)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    
    if scheduler is not None:
        scheduler.step()
    
    return total_loss / len(train_data_list)


def evaluate(model, data_list, edge_index, edge_weight, labels, criterion, device):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    total_loss = 0
    
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device) if edge_weight is not None else None
    
    with torch.no_grad():
        for data, label in zip(data_list, labels):
            x = data.x.to(device)
            pe = data.pe.to(device)
            se = data.se.to(device)
            lambda_max = data.lambda_max.to(device)
            y = torch.tensor([label], dtype=torch.long).to(device)
            
            logits, _, _ = model(x, edge_index, pe, se, edge_weight, lambda_max)
            loss = criterion(logits, y)
            total_loss += loss.item()
            
            probs = F.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1)
            
            all_labels.append(label)
            all_preds.append(pred.item())
            all_probs.append(probs[:, 1].item())
    
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    
    metrics = {
        'loss': total_loss / len(data_list),
        'accuracy': accuracy_score(all_labels, all_preds),
        'balanced_accuracy': balanced_accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, average='macro'),
        'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'mcc': matthews_corrcoef(all_labels, all_preds),
        'auc': roc_auc_score(all_labels, all_probs),
        'auprc': average_precision_score(all_labels, all_probs),
        'labels': all_labels,
        'preds': all_preds,
        'probs': all_probs
    }
    return metrics


def extract_genes(model, sample_data, edge_index, edge_weight, nodes, device, top_k=50):

    model.eval()
    with torch.no_grad():
        x = sample_data.x.to(device)
        pe = sample_data.pe.to(device)
        se = sample_data.se.to(device)
        lambda_max = sample_data.lambda_max.to(device)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device) if edge_weight is not None else None
        
        _, importance, _ = model(x, edge_index, pe, se, edge_weight, lambda_max)
        scores = importance.cpu().numpy()
        
        top_idx = np.argsort(scores)[::-1][:top_k]
        top_genes = nodes.iloc[top_idx]['gene'].tolist()
        
        return top_genes, scores[top_idx], scores

def plot_training_curves(history, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    epochs = range(1, len(history['train_loss']) + 1)
    
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training & Validation Loss'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, history['val_auc'], 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('AUC-ROC')
    axes[0, 1].set_title('Validation AUC-ROC'); axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(epochs, history['val_auprc'], 'm-', linewidth=2)
    axes[0, 2].set_xlabel('Epoch'); axes[0, 2].set_ylabel('AUPRC')
    axes[0, 2].set_title('Validation AUPRC'); axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].plot(epochs, history['val_acc'], 'c-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Validation Accuracy'); axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(epochs, history['val_f1'], 'orange', linewidth=2)
    axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('Validation F1 (macro)'); axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].plot(epochs, history['val_mcc'], 'purple', linewidth=2)
    axes[1, 2].set_xlabel('Epoch'); axes[1, 2].set_ylabel('MCC')
    axes[1, 2].set_title('Validation Matthews Correlation'); axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"训练曲线已保存: {save_path}")


def plot_roc_pr_curves(labels, probs, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    axes[0].plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc:.4f}')
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
    axes[0].set_title('ROC Curve'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    
    precision, recall, _ = precision_recall_curve(labels, probs)
    auprc = average_precision_score(labels, probs)
    axes[1].plot(recall, precision, 'r-', linewidth=2, label=f'AUPRC = {auprc:.4f}')
    axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"ROC/PR曲线已保存: {save_path}")


def plot_confusion_matrix(labels, preds, save_path):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Adenoma', 'Cancer'], yticklabels=['Adenoma', 'Cancer'])
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"混淆矩阵已保存: {save_path}")


def plot_gene_importance(top_genes, scores, all_scores, nodes, save_path, top_n=20):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(top_genes)))
    axes[0].barh(range(len(top_genes)), scores[::-1], color=colors[::-1])
    axes[0].set_yticks(range(len(top_genes)))
    axes[0].set_yticklabels(top_genes[::-1], fontsize=10)
    axes[0].set_xlabel('Importance Score')
    axes[0].set_title(f'Top-{top_n} Candidate Driver Genes')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    axes[1].hist(all_scores, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[1].axvline(x=np.mean(all_scores), color='r', linestyle='--', label=f'Mean: {np.mean(all_scores):.4f}')
    axes[1].axvline(x=np.median(all_scores), color='g', linestyle='--', label=f'Median: {np.median(all_scores):.4f}')
    axes[1].set_xlabel('Importance Score'); axes[1].set_ylabel('Frequency')
    axes[1].set_title('Gene Importance Distribution'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"基因重要性图已保存: {save_path}")


def plot_cross_validation_results(cv_results, save_path):
    metrics_names = ['accuracy', 'balanced_accuracy', 'f1', 'auc', 'auprc', 'mcc']
    metrics_data = [[r[m] for r in cv_results] for m in metrics_names]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(metrics_data, labels=[m.replace('_', ' ').title() for m in metrics_names],
                    patch_artist=True, showmeans=True)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(metrics_names)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Score'); ax.set_title('Cross-Validation Results'); ax.grid(True, alpha=0.3, axis='y')
    
    for i, metric in enumerate(metrics_names):
        mean_val = np.mean([r[metric] for r in cv_results])
        ax.text(i+1, mean_val+0.02, f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"交叉验证结果已保存: {save_path}")


def train_single_fold(train_data, val_data, test_data, train_labels, val_labels, test_labels,
                      edge_index, edge_weight, device, fold_idx=0):

    in_dim = train_data[0].x.shape[1]
    
    model = ChebNetII_Transformer_PESE(
        in_dim=in_dim, 
        hidden_dim=128, 
        out_dim=128,
        pe_k=8, 
        se_walk=20, 
        K=5, 
        num_heads=4,
        num_transformer_layers=2, 
        dropout=0.3
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=20)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=180, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[20])
    

    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.05)
    
    history = {
        'train_loss': [], 'val_loss': [], 'val_auc': [], 
        'val_auprc': [], 'val_acc': [], 'val_f1': [], 'val_mcc': []
    }
    
    best_auc = 0
    patience_counter = 0
    best_state = None
    best_epoch = 0
    
    pbar = tqdm(range(200), desc=f'Fold {fold_idx} Epoch', leave=False)
    for epoch in pbar:
        train_loss = train_epoch(model, train_data, edge_index, edge_weight, train_labels, 
                                 optimizer, criterion, device, scheduler)
        
        val_metrics = evaluate(model, val_data, edge_index, edge_weight, val_labels, 
                               criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_auc'].append(val_metrics['auc'])
        history['val_auprc'].append(val_metrics['auprc'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_mcc'].append(val_metrics['mcc'])
        
        pbar.set_postfix({
            'loss': f'{train_loss:.3f}',
            'AUC': f'{val_metrics["auc"]:.3f}',
            'F1': f'{val_metrics["f1"]:.3f}'
        })
        
        if (epoch + 1) % 20 == 0:
            print(f"Fold {fold_idx} Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | "
                  f"Val AUC: {val_metrics['auc']:.4f} | Val F1: {val_metrics['f1']:.4f}")
        
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_state = model.state_dict().copy()
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 30:
                pbar.set_postfix_str(f'Early stop at epoch {epoch+1}, best AUC={best_auc:.4f}')
                pbar.close()
                print(f"Fold {fold_idx} 早停于 Epoch {epoch+1}, 最佳AUC: {best_auc:.4f} (Epoch {best_epoch+1})")
                break
    
    if best_state:
        model.load_state_dict(best_state)
    
    test_metrics = evaluate(model, test_data, edge_index, edge_weight, test_labels, criterion, device)
    return model, test_metrics, history, best_epoch


def main(use_cv=True, n_splits=5):
    print("=" * 70)
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}\n")
    
    output_dir = "results_chebnetii_transformer_pese"
    os.makedirs(output_dir, exist_ok=True)
    
    print("[1/5] 加载数据...")
    sample_labels = load_sample_labels()
    expression_df = load_expression_data()
    edges, nodes = load_ppi_network()
    
    common = set(expression_df.columns) & set(sample_labels['Sample_ID'])
    sample_labels = sample_labels[sample_labels['Sample_ID'].isin(common)]
    print(f"共同样本: {len(common)}")
    
    labels = sample_labels['label'].values
    sample_ids = sample_labels['Sample_ID'].values
    
    print("\n[2/5] 构建图数据（含PE/SE）...")
    label_dict = dict(zip(sample_labels['Sample_ID'], sample_labels['label']))

    all_data, L, lambda_max, edge_index, edge_weight = prepare_data(
        sample_ids, expression_df, edges, nodes, label_dict, pe_k=8, se_walk=20
    )
    
    if use_cv:
        print(f"\n[3/5] 进行 {n_splits} 折交叉验证 (6:2:2 划分)...")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        
        cv_results = []
        all_histories = []
        all_gene_scores = []
        
        for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(sample_ids, labels)):
            print(f"\n{'='*50}")
            print(f"Fold {fold_idx + 1}/{n_splits}")
            print('='*50)
            
            if n_splits == 5:
                train_size = 0.75  
            elif n_splits == 4:
                train_size = 0.8   
            elif n_splits == 3:
                train_size = 0.9   
            else:
                train_size = 0.75
            
            train_idx, val_idx = train_test_split(
                train_val_idx, train_size=train_size, stratify=labels[train_val_idx], random_state=SEED
            )
            
            train_data = [all_data[i] for i in train_idx]
            val_data = [all_data[i] for i in val_idx]
            test_data = [all_data[i] for i in test_idx]
            
            train_labels, val_labels, test_labels = labels[train_idx], labels[val_idx], labels[test_idx]
            
            print(f"训练: {len(train_idx)} | 验证: {len(val_idx)} | 测试: {len(test_idx)}")
            print(f"比例: {len(train_idx)/len(all_data):.1%} : {len(val_idx)/len(all_data):.1%} : {len(test_idx)/len(all_data):.1%}")
            
            model, test_metrics, history, best_epoch = train_single_fold(
                train_data, val_data, test_data, train_labels, val_labels, test_labels,
                edge_index, edge_weight, device, fold_idx + 1
            )
            
            cv_results.append(test_metrics)
            all_histories.append(history)
            
            print(f"\nFold {fold_idx+1} 测试结果:")
            for k in ['accuracy', 'balanced_accuracy', 'f1', 'auc', 'auprc', 'mcc']:
                print(f"  {k}: {test_metrics[k]:.4f}")
            
            _, _, gene_scores = extract_genes(model, test_data[0], edge_index, edge_weight, nodes, device, 50)
            all_gene_scores.append(gene_scores)
        
        print("\n" + "=" * 70)
        print("交叉验证汇总结果 (Mean ± Std)")
        print("=" * 70)
        
        for metric in ['accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall', 
                       'sensitivity', 'specificity', 'mcc', 'auc', 'auprc']:
            values = [r[metric] for r in cv_results]
            print(f"  {metric:20s}: {np.mean(values):.4f} ± {np.std(values):.4f}")
        
        plot_cross_validation_results(cv_results, f"{output_dir}/cv_results.png")
        
        best_fold = np.argmax([r['auc'] for r in cv_results])
        best_history = all_histories[best_fold]
        best_metrics = cv_results[best_fold]
        
        print(f"\n使用 Fold {best_fold+1} (最佳AUC) 进行详细可视化...")
        
    else:
        print("\n[3/5] 划分数据 (6:2:2)...")
        train_idx, temp = train_test_split(range(len(sample_ids)), train_size=0.6, stratify=labels, random_state=SEED)
        val_idx, test_idx = train_test_split(temp, train_size=0.5, stratify=labels[temp], random_state=SEED)
        
        train_data = [all_data[i] for i in train_idx]
        val_data = [all_data[i] for i in val_idx]
        test_data = [all_data[i] for i in test_idx]
        
        train_labels, val_labels, test_labels = labels[train_idx], labels[val_idx], labels[test_idx]
        
        print(f"训练: {len(train_idx)} | 验证: {len(val_idx)} | 测试: {len(test_idx)}")
        
        model, best_metrics, best_history, best_epoch = train_single_fold(
            train_data, val_data, test_data, train_labels, val_labels, test_labels,
            edge_index, edge_weight, device
        )
        
        all_gene_scores = []
        _, _, gene_scores = extract_genes(model, test_data[0], edge_index, edge_weight, nodes, device, 50)
        all_gene_scores.append(gene_scores)
    
    print("\n[4/5] 生成可视化...")
    plot_training_curves(best_history, f"{output_dir}/training_curves.png")
    plot_roc_pr_curves(best_metrics['labels'], best_metrics['probs'], f"{output_dir}/roc_pr_curves.png")
    plot_confusion_matrix(best_metrics['labels'], best_metrics['preds'], f"{output_dir}/confusion_matrix.png")
    
    avg_scores = np.mean(all_gene_scores, axis=0) if len(all_gene_scores) > 1 else all_gene_scores[0]
    top_idx = np.argsort(avg_scores)[::-1][:50]
    top_genes = nodes.iloc[top_idx]['gene'].tolist()
    top_scores = avg_scores[top_idx]
    
    plot_gene_importance(top_genes, top_scores, avg_scores, nodes, f"{output_dir}/gene_importance.png", top_n=10)
    
    print("\n" + "=" * 50)
    print("Top-10 候选驱动基因:")
    for i, (gene, score) in enumerate(zip(top_genes[:20], top_scores[:20])):
        print(f"  {i+1:2d}. {gene:15s} (importance: {score:.6f})")
    
    torch.save({
        'model': model.state_dict() if not use_cv else None,
        'metrics': best_metrics,
        'top_genes': top_genes,
        'cv_results': cv_results if use_cv else None
    }, f"{output_dir}/model_results.pt")
    
    print(f"\n 所有结果已保存至: {output_dir}/")


if __name__ == "__main__":
    main(use_cv=True, n_splits=5)