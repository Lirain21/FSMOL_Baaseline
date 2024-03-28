from dataclasses import dataclass
from typing import List, Tuple
from typing_extensions import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from fs_mol.modules.graph_feature_extractor import (
    GraphFeatureExtractor,
    GraphFeatureExtractorConfig,
)
# from fs_mol.data.dkt import DKTBatch
from fs_mol.data.prw_dkt import DKTBatch 

#from fs_mol.utils._stateless import functional_call

import sys
sys.path.append("./PAR-NeurIPS21/")
sys.path.append("../../PAR-NeurIPS21/")
from chem_lib.models.mol_model import attention, ContextAwareRelationNet
from chem_lib.models.relation import MLP, ContextMLP, TaskAwareRelation

FINGERPRINT_DIM = 2048
PHYS_CHEM_DESCRIPTORS_DIM = 42


@dataclass(frozen=True)
class PRWModelConfig:
    # Model configuration:
    graph_feature_extractor_config: GraphFeatureExtractorConfig = GraphFeatureExtractorConfig()
    used_features: Literal[
        "gnn", "ecfp", "pc-descs", "gnn+ecfp", "ecfp+fc", "pc-descs+fc", "gnn+ecfp+pc-descs+fc"
    ] = "gnn+ecfp+fc"
    #distance_metric: Literal["mahalanobis", "euclidean"] = "mahalanobis"


class PRWModel(nn.Module):
    def __init__(self, config: PRWModelConfig):
        super().__init__()
        self.config = config
        # Create GNN if needed:
        if self.config.used_features.startswith("gnn"):
            self.graph_feature_extractor = GraphFeatureExtractor(
                config.graph_feature_extractor_config
            )

        self.use_fc = self.config.used_features.endswith("+fc")

        # Create MLP if needed:
        if self.use_fc:
            # Determine dimension:
            fc_in_dim = 0
            if "gnn" in self.config.used_features:
                fc_in_dim += self.config.graph_feature_extractor_config.readout_config.output_dim
            if "ecfp" in self.config.used_features:
                fc_in_dim += FINGERPRINT_DIM
            if "pc-descs" in self.config.used_features:
                fc_in_dim += PHYS_CHEM_DESCRIPTORS_DIM

            self.enc_fc = nn.Sequential(
                nn.Linear(fc_in_dim, 512),
                nn.ReLU(),
                nn.Linear(512, self.config.emb_dim),
            )

        inp_dim = self.config.emb_dim
        # self.encode_projection = ContextMLP(inp_dim=self.config.emb_dim, hidden_dim=self.config.map_dim, num_layers=self.config.map_layer,
        #                         batch_norm=self.config.batch_norm,dropout=self.config.map_dropout,
        #                         pre_fc=self.config.map_pre_fc,ctx_head=self.config.ctx_head)
        self.mlp_proj = MLP(inp_dim=inp_dim, hidden_dim=self.config.map_dim, num_layers=self.config.map_layer,
                batch_norm=self.config.batch_norm, dropout=self.config.map_dropout)

    def to_one_hot(self,class_idx, num_classes=2):
        return torch.eye(num_classes)[class_idx].to(class_idx.device)
    
    def get_proto(self, s_emb, unlabeled_emb):
        n_support, n_dim = s_emb.shape
        n_shot = int(n_support // 2)
        label = [0 for i in range(n_shot)]
        label.extend([1 for i in range(n_shot)])
        label = torch.tensor(label).to(s_emb.device)
        n_unlabeled = unlabeled_emb.size(0)
        label_support_onehot = F.one_hot(label, 2)
        neg_proto_emb = s_emb[:n_shot,:].mean(0)
        pos_proto_emb = s_emb[n_shot:, :].mean(0)
        proto = torch.stack((neg_proto_emb, pos_proto_emb), dim=0)
        n, d = proto.shape 
        dis = -torch.cdist(unlabeled_emb, proto)
        z_hat = F.softmax(dis, dim=1)
        z = torch.cat((label_support_onehot, z_hat), dim=0)
        h = torch.cat((s_emb, unlabeled_emb), dim=0)
        proto = torch.bmm(z.transpose(0,1).unsqueeze(0), h.unsqueeze(0)).squeeze(0)
        sum_z = z.sum(dim=0).unsqueeze(1).repeat(1, n_dim)
        proto = proto / sum_z 
        return proto 
        
        

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def relation_forward(self, s_emb, q_emb, s_label=None, q_pred_adj=False,return_adj=False,return_emb=False):
        if not return_emb:
            s_logits, q_logits, adj = self.adapt_relation(s_emb, q_emb,return_adj=return_adj,return_emb=return_emb)
        else:
            s_logits, q_logits, adj, s_rel_emb, q_rel_emb = self.adapt_relation(s_emb, q_emb,return_adj=return_adj,return_emb=return_emb)
        if q_pred_adj:
            q_sim = adj[-1][:, 0, -1, :-1]
            q_logits = q_sim @ self.to_one_hot(s_label)
        if not return_emb:
            return s_logits, q_logits, adj
        else:
            return s_logits, q_logits, adj, s_rel_emb, q_rel_emb
        
    
    def _euclidean_distances(
        self, query_features: torch.Tensor, class_prototypes: torch.Tensor
    ) -> torch.Tensor:
        num_query_features = query_features.shape[0]
        num_prototypes = class_prototypes.shape[0]

        distances = (
            (
                query_features.unsqueeze(1).expand(num_query_features, num_prototypes, -1)
                - class_prototypes.unsqueeze(0).expand(num_query_features, num_prototypes, -1)
            )
            .pow(2)
            .sum(dim=2)
        )

        return -distances

    def forward(self, input_batch: DKTBatch, q_pred_adj=False):
        support_features: List[torch.Tensor] = []
        query_features: List[torch.Tensor] = []
        unlabeled_features: List[torch.Tensor] = []

        if "gnn" in self.config.used_features:
            support_features.append(self.graph_feature_extractor(input_batch.support_features))
            query_features.append(self.graph_feature_extractor(input_batch.query_features))
            unlabeled_features.append(self.graph_feature_extractor(input_batch.unlabeled_features))
        if "ecfp" in self.config.used_features:
            support_features.append(input_batch.support_features.fingerprints.float())
            query_features.append(input_batch.query_features.fingerprints.float())
            unlabeled_features.append(input_batch.unlabeled_features.fingerprints.float())
        if "pc-descs" in self.config.used_features:
            support_features.append(input_batch.support_features.descriptors)
            query_features.append(input_batch.query_features.descriptors)
            unlabeled_features.append(input_batch.unlabeled_features.descriptors)

        support_features_flat = torch.cat(support_features, dim=1)
        query_features_flat = torch.cat(query_features, dim=1)
        unlabeled_features_flat = torch.cat(unlabeled_features, dim=1)

        if self.use_fc:
            support_features_flat = self.enc_fc(support_features_flat)
            query_features_flat = self.enc_fc(query_features_flat)
            unlabeled_features_flat = self.enc_fc(unlabeled_features_flat)

        if self.config.use_numeric_labels:
            raise NotImplementedError
        else:
            support_labels_converted = input_batch.support_labels
            query_labels_converted = input_batch.query_labels
            unlabeled_labels_converted = input_batch.unlabeled_labels 

        refined_proto = self.get_proto(support_features_flat, unlabeled_features_flat)
        
        support_features_flat = self.mlp_proj(support_features_flat)
        query_features_flat = self.mlp_proj(query_features_flat)
        refined_proto = self.mlp_proj(refined_proto)
        
        # s_logits = F.softmax(-torch.cdist(support_features_flat, refined_proto.detach()), dim=-1)
        # q_logits = F.softmax(-torch.cdist(query_features_flat, refined_proto.detach()), dim=-1)
        
        s_logits = self._euclidean_distances(support_features_flat, refined_proto)
        q_logits = self._euclidean_distances(query_features_flat, refined_proto)
        
        return s_logits, q_logits
