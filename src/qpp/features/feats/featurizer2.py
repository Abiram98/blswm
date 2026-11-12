"""
Featurizer2 - Unified feature vector architecture following PlanRGCN philosophy.

This featurizer reuses TriplePattern (TP) features in PathNode (PP) features,
making it more efficient with a unified feature vector where:
- TP nodes use: base_features + zeros_for_path_ops
- PP nodes use: base_features + path_operations + pred_min_max
- Total dimensions: base_size + path_size + filter_size

This matches PlanRGCN's approach where TP features are a subset of PP features.
"""

import json
import os
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

from qpp.features.nodes.path_node import PathNode
from qpp.features.qp.qp_utils import pathOpTypes
from qpp.features.qp.visitor.UtilVisitor import LiteralsFeaturizer
from qpp.features.nodes.node import FilterNode, TriplePattern, TriplePattern2, TriplePattern3


class Featurizer2:
    """
    Unified featurizer where TP features are reused in PP features (PlanRGCN philosophy).
    
    Architecture:
    - base_size: Common features for both TP and PP (var_vec + pred + ent*2 + lit)
    - path_size: Path-specific features (path_ops + pred_min_max)
    - filter_size: Filter node features
    
    Total feature vector: [base_features | path_features | filter_features]
    
    For TP nodes: path_features are zeros
    For PP nodes: path_features are populated
    For Filter nodes: base_features and path_features are zeros
    """
    
    def __init__(
        self,
        pred_stat_path="",
        ent_path="",
        lit_path=None,
        bins=10,
        scaling="binner",
    ):
        """
        Initialize the featurizer with statistical data paths.
        
        Args:
            pred_stat_path: Path to predicate statistics directory
            ent_path: Path to entity statistics directory  
            lit_path: Path to literal statistics directory (optional)
            bins: Number of bins for quantile-based discretization
            scaling: Scaling strategy ('binner' or 'None')
        """
        self.pred_stat_path = pred_stat_path
        self.ent_path = ent_path
        self.lit_path = lit_path
        self.bins = bins
        self.scaling = scaling
        
        # Load predicate statistics
        self.pred_freq = {}
        self.pred_ents = {}
        self.pred_lits = {}
        self._load_pred_stats()
        
        # Load entity statistics
        self.ent_freq = {}
        self.ent_subj = {}
        self.ent_obj = {}
        self._load_ent_stats()
        
        # Load literal statistics if path provided
        self.lit_freq = {}
        if lit_path is not None:
            self._load_lit_stats()
        
        # Initialize scalers based on scaling strategy
        self.filter_size = 6
        self._initialize_scalers()
        
        # Calculate feature vector sizes
        self._calculate_sizes()
        
        # Create masks for different node types
        # TP: base features only, path features are zeros
        self.tp_mask = [1] * self.base_size + [0] * self.path_size + [0] * self.filter_size
        # PP: base + path features, filter is zeros
        self.pp_mask = [1] * self.base_size + [1] * self.path_size + [0] * self.filter_size
        # Filter: only filter features
        self.filter_mask = [0] * self.base_size + [0] * self.path_size + [1] * self.filter_size
        
        # For compatibility with existing code that uses tp_size
        self.tp_size = self.base_size + self.path_size + self.filter_size
    
    def _load_pred_stats(self):
        """Load predicate statistics from JSON files."""
        freq_path = os.path.join(self.pred_stat_path, "freq/")
        ent_path = os.path.join(self.pred_stat_path, "ents/")
        lits_path = os.path.join(self.pred_stat_path, "lits/")
        
        if not os.path.exists(freq_path):
            raise Exception(f"Predicate feature path not existing: {freq_path}")
        
        # Load frequency stats
        for file in sorted([f for f in os.listdir(freq_path) if f.endswith(".json")]):
            self._load_pred_freq_file(os.path.join(freq_path, file))
        
        # Load entity stats
        if os.path.exists(ent_path):
            for file in sorted([f for f in os.listdir(ent_path) if f.endswith(".json")]):
                self._load_pred_ents_file(os.path.join(ent_path, file))
        
        # Load literal stats
        if os.path.exists(lits_path):
            for file in sorted([f for f in os.listdir(lits_path) if f.endswith(".json")]):
                self._load_pred_lits_file(os.path.join(lits_path, file))
    
    def _load_pred_freq_file(self, filepath):
        """Load predicate frequency from a single JSON file."""
        data = json.load(open(filepath, "r"))
        bindings = data.get("results", {}).get("bindings", [])
        
        for item in bindings:
            if "p1" in item and "triples" in item:
                pred = item["p1"]["value"]
                count = item["triples"]["value"]
                self.pred_freq[pred] = count
    
    def _load_pred_ents_file(self, filepath):
        """Load predicate entity counts from a single JSON file."""
        data = json.load(open(filepath, "r"))
        bindings = data.get("results", {}).get("bindings", [])
        
        for item in bindings:
            if "p1" in item and "entities" in item:
                pred = item["p1"]["value"]
                count = item["entities"]["value"]
                self.pred_ents[pred] = count
    
    def _load_pred_lits_file(self, filepath):
        """Load predicate literal counts from a single JSON file."""
        data = json.load(open(filepath, "r"))
        bindings = data.get("results", {}).get("bindings", [])
        
        for item in bindings:
            if "p1" in item and "literals" in item:
                pred = item["p1"]["value"]
                count = item["literals"]["value"]
                self.pred_lits[pred] = count
    
    def _load_ent_stats(self):
        """Load entity statistics from JSON files."""
        freq_path = os.path.join(self.ent_path, "freq/")
        subj_path = os.path.join(self.ent_path, "subj/")
        obj_path = os.path.join(self.ent_path, "obj/")
        
        if not os.path.exists(freq_path):
            raise Exception(f"Entity feature path not existing: {freq_path}")
        
        # Load frequency stats
        for file in sorted([f for f in os.listdir(freq_path) if f.endswith(".json")]):
            self._load_ent_freq_file(os.path.join(freq_path, file))
        
        # Load subject stats
        if os.path.exists(subj_path):
            for file in sorted([f for f in os.listdir(subj_path) if f.endswith(".json")]):
                self._load_ent_subj_file(os.path.join(subj_path, file))
        
        # Load object stats
        if os.path.exists(obj_path):
            for file in sorted([f for f in os.listdir(obj_path) if f.endswith(".json")]):
                self._load_ent_obj_file(os.path.join(obj_path, file))
    
    def _load_ent_freq_file(self, filepath):
        """Load entity frequency from a single JSON file."""
        data = json.load(open(filepath, "r"))
        bindings = data.get("results", {}).get("bindings", [])
        
        for item in bindings:
            if "e" in item and "entities" in item:
                ent = item["e"]["value"]
                count = item["entities"]["value"]
                self.ent_freq[ent] = count
    
    def _load_ent_subj_file(self, filepath):
        """Load entity subject position counts from a single JSON file."""
        data = json.load(open(filepath, "r"))
        bindings = data.get("results", {}).get("bindings", [])
        
        for item in bindings:
            if "e" in item and "entities" in item:
                ent = item["e"]["value"]
                count = item["entities"]["value"]
                self.ent_subj[ent] = count
    
    def _load_ent_obj_file(self, filepath):
        """Load entity object position counts from a single JSON file."""
        data = json.load(open(filepath, "r"))
        bindings = data.get("results", {}).get("bindings", [])
        
        for item in bindings:
            if "e" in item and "entities" in item:
                ent = item["e"]["value"]
                count = item["entities"]["value"]
                self.ent_obj[ent] = count
    
    def _load_lit_stats(self):
        """Load literal statistics from JSON files."""
        freq_path = os.path.join(self.lit_path, "freq/")
        
        if not os.path.exists(freq_path):
            raise Exception(f"Literal feature path not existing: {freq_path}")
        
        for file in sorted([f for f in os.listdir(freq_path) if f.endswith(".json")]):
            self._load_lit_freq_file(os.path.join(freq_path, file))
    
    def _load_lit_freq_file(self, filepath):
        """Load literal frequency from a single JSON file."""
        data = json.load(open(filepath, "r"))
        bindings = data.get("results", {}).get("bindings", [])
        
        for item in bindings:
            if "e" in item and "entities" in item:
                lit = item["e"]["value"]
                count = item["entities"]["value"]
                # Take maximum if duplicate
                if lit in self.lit_freq:
                    self.lit_freq[lit] = max(self.lit_freq[lit], count)
                else:
                    self.lit_freq[lit] = count
    
    def _initialize_scalers(self):
        """Initialize scalers based on the chosen scaling strategy."""
        if self.scaling == "binner":
            self._init_binner_scalers()
        elif self.scaling == "None":
            # No scaling, but we still need to track feature sizes
            self.pred_feat_size = 3
            self.ent_feat_size = 3
            self.lit_feat_size = 1 if self.lit_path else 0
        else:
            raise ValueError(f"Unsupported scaling option: {self.scaling}")
    
    def _init_binner_scalers(self):
        """Initialize quantile-based binning scalers."""
        # Predicate frequency scaler
        self.pred_freq_scaler = KBinsDiscretizer(
            n_bins=self.bins,
            strategy="quantile",
            encode="onehot-dense",
            random_state=42,
        )
        pred_freq_vals = np.array(sorted([int(x) for x in self.pred_freq.values()])).reshape(-1, 1)
        self.pred_freq_scaler.fit(pred_freq_vals)
        
        # Predicate literal count scaler
        self.pred_lits_scaler = KBinsDiscretizer(
            n_bins=self.bins,
            strategy="quantile",
            encode="onehot-dense",
            random_state=42,
        )
        pred_lits_vals = np.array(sorted([int(x) for x in self.pred_lits.values()])).reshape(-1, 1)
        self.pred_lits_scaler.fit(pred_lits_vals)
        
        # Predicate entity count scaler
        self.pred_ents_scaler = KBinsDiscretizer(
            n_bins=self.bins,
            strategy="quantile",
            encode="onehot-dense",
            random_state=42,
        )
        pred_ents_vals = np.array(sorted([int(x) for x in self.pred_ents.values()])).reshape(-1, 1)
        self.pred_ents_scaler.fit(pred_ents_vals)
        
        # Entity frequency scaler
        self.ent_freq_scaler = KBinsDiscretizer(
            n_bins=self.bins,
            strategy="quantile",
            encode="onehot-dense",
            random_state=42,
        )
        ent_freq_vals = np.array(sorted([int(x) for x in self.ent_freq.values()])).reshape(-1, 1)
        self.ent_freq_scaler.fit(ent_freq_vals)
        
        # Entity subject position scaler
        self.ent_subj_scaler = KBinsDiscretizer(
            n_bins=self.bins,
            strategy="quantile",
            encode="onehot-dense",
            random_state=42,
        )
        ent_subj_vals = np.array(sorted([int(x) for x in self.ent_subj.values()])).reshape(-1, 1)
        self.ent_subj_scaler.fit(ent_subj_vals)
        
        # Entity object position scaler
        self.ent_obj_scaler = KBinsDiscretizer(
            n_bins=self.bins,
            strategy="quantile",
            encode="onehot-dense",
            random_state=42,
        )
        ent_obj_vals = np.array(sorted([int(x) for x in self.ent_obj.values()])).reshape(-1, 1)
        self.ent_obj_scaler.fit(ent_obj_vals)
        
        # Literal frequency scaler (if literals are used)
        if self.lit_path is not None:
            self.lit_freq_scaler = KBinsDiscretizer(
                n_bins=self.bins,
                strategy="quantile",
                encode="onehot-dense",
                random_state=42,
            )
            lit_freq_vals = np.array(sorted([int(x) for x in self.lit_freq.values()])).reshape(-1, 1)
            self.lit_freq_scaler.fit(lit_freq_vals)
    
    def _calculate_sizes(self):
        """
        Calculate feature vector sizes using unified architecture.
        
        Base features (shared by TP and PP):
            - var_vec (3)
            - predicate stats (binned)
            - entity stats * 2 (binned, for subject and object)
            - literal stats (optional, binned)
            - literal types (optional, 7 features)
        
        Path features (only for PP):
            - path operations (7)
            - pred_min_max (2)
        """
        if self.scaling == "binner":
            # Predicate features: freq + lits + ents (each binned)
            pred_scale_len = (
                self.pred_freq_scaler.n_bins_[0]
                + self.pred_lits_scaler.n_bins_[0]
                + self.pred_ents_scaler.n_bins_[0]
            )
            
            # Entity features: freq + subj + obj (each binned)
            ent_scale_len = (
                self.ent_freq_scaler.n_bins_[0]
                + self.ent_subj_scaler.n_bins_[0]
                + self.ent_obj_scaler.n_bins_[0]
            )
            
            # Base features (shared by TP and PP nodes)
            if self.lit_path is not None:
                lit_scale_len = self.lit_freq_scaler.n_bins_[0]
                lit_type_len = LiteralsFeaturizer.feat_size()
                
                self.base_size = (
                    3  # var_vec
                    + pred_scale_len
                    + ent_scale_len * 2  # subject and object
                    + lit_scale_len
                    + lit_type_len
                )
            else:
                self.base_size = (
                    3  # var_vec
                    + pred_scale_len
                    + ent_scale_len * 2
                )
            
            # Path-specific features (only for PP nodes)
            self.path_size = (
                pathOpTypes.get_max_operations()  # 7
                + 2  # pred_min_max
            )
            
        else:  # scaling == "None"
            if self.lit_path is not None:
                self.base_size = 3 + 3 + 6 + 1 + LiteralsFeaturizer.feat_size()
            else:
                self.base_size = 3 + 3 + 6
            
            self.path_size = pathOpTypes.get_max_operations() + 2
    
    def get_value_dict(self, dct, key):
        """
        Safely retrieve a value from a dictionary, stripping angle brackets from keys.
        
        Args:
            dct: Dictionary to query
            key: Key to look up (may have angle brackets)
            
        Returns:
            Value from dictionary or 0 if not found
        """
        try:
            # Strip angle brackets if present
            if key.startswith('<'):
                key = key[1:]
            if key.endswith('>'):
                key = key[:-1]
            return dct[key]
        except KeyError:
            return 0
    
    def featurize(self, node):
        """
        Main featurization method that routes to appropriate feature extractor.
        
        Args:
            node: Query graph node (FilterNode, PathNode, TriplePattern2, or TriplePattern3)
            
        Returns:
            numpy array of features
        """
        if isinstance(node, FilterNode):
            return self.filter_features(node).astype("float32")
        elif isinstance(node, PathNode):
            return self.pp_features(node).astype("float32")
        elif isinstance(node, (TriplePattern2, TriplePattern3, TriplePattern)):
            return self.tp_features(node).astype("float32")
        else:
            raise Exception(f"Unknown node type: {type(node)}")
    
    def filter_features(self, node):
        """
        Extract features from a filter node.
        
        Args:
            node: FilterNode instance
            
        Returns:
            Feature vector: [zeros_base | zeros_path | filter_features]
        """
        op_cats = {
            "ARITHMETIC": 0,
            "LOGICAL": 1,
            "COMPARISON": 2,
            "GENERAL": 3,
            "STRING": 4,
            "TIME": 5
        }
        vec = np.zeros(6)
        for k in node.operationTypes:
            vec[op_cats[k]] = 1
        
        # Filter nodes: base and path features are zeros
        return np.concatenate((
            np.zeros(self.base_size),
            np.zeros(self.path_size),
            np.array(vec)
        ), axis=0)
    
    def tp_features(self, node):
        """
        Extract features from a triple pattern node.
        
        Args:
            node: TriplePattern2 instance
            
        Returns:
            Feature vector: [base_features | zeros_path | zeros_filter]
        """
        base_vec = self._extract_base_features(node)
        
        # TP nodes: base features + zeros for path features
        return np.concatenate((
            base_vec,
            np.zeros(self.path_size),
            np.zeros(self.filter_size)
        ), axis=0)
    
    def pp_features(self, node):
        """
        Extract features from a property path node.
        
        Args:
            node: PathNode instance
            
        Returns:
            Feature vector: [base_features | path_features | zeros_filter]
        """
        base_vec = self._extract_base_features(node)
        
        # Path-specific features
        path_operation = np.zeros(pathOpTypes.get_max_operations())
        pred_min_max = np.zeros(2)
        
        if isinstance(node, PathNode):
            pred_min_max[0] = node.p_mod_min
            pred_min_max[1] = node.p_mod_max
            for op in node.path_complexity:
                path_operation[op.value] = 1
        
        path_vec = np.concatenate([path_operation, pred_min_max])
        
        # PP nodes: base features + path features + zeros for filter
        return np.concatenate((
            base_vec,
            path_vec,
            np.zeros(self.filter_size)
        ), axis=0)
    
    def _extract_base_features(self, node):
        """
        Extract base features that are shared between TP and PP nodes.
        
        Args:
            node: TriplePattern2 or PathNode instance
            
        Returns:
            Base feature vector (var_vec + pred + ent*2 + lit)
        """
        # Variable type vector
        var_vec = np.array([
            node.subject.nodetype,
            node.predicate.nodetype,
            node.object.nodetype
        ])
        
        # Predicate statistics
        pred_freq = self.get_value_dict(self.pred_freq, node.predicate.node_label)
        pred_lits = self.get_value_dict(self.pred_lits, node.predicate.node_label)
        pred_ents = self.get_value_dict(self.pred_ents, node.predicate.node_label)
        
        if self.scaling == "binner":
            pred_freq = self.pred_freq_scaler.transform([[pred_freq]])[0]
            pred_lits = self.pred_lits_scaler.transform([[pred_lits]])[0]
            pred_ents = self.pred_ents_scaler.transform([[pred_ents]])[0]
        
        # Subject entity statistics
        if node.subject.type == "URI":
            subj_freq = self.get_value_dict(self.ent_freq, node.subject.node_label)
            subj_subj_freq = self.get_value_dict(self.ent_subj, node.subject.node_label)
            sub_obj_freq = self.get_value_dict(self.ent_obj, node.subject.node_label)
            
            if self.scaling == "binner":
                subj_freq = self.ent_freq_scaler.transform([[subj_freq]])[0]
                subj_subj_freq = self.ent_subj_scaler.transform([[subj_subj_freq]])[0]
                sub_obj_freq = self.ent_obj_scaler.transform([[sub_obj_freq]])[0]
        else:
            if self.scaling == "binner":
                subj_freq = np.zeros(self.ent_freq_scaler.n_bins_[0])
                subj_subj_freq = np.zeros(self.ent_subj_scaler.n_bins_[0])
                sub_obj_freq = np.zeros(self.ent_obj_scaler.n_bins_[0])
            else:
                subj_freq, subj_subj_freq, sub_obj_freq = 0, 0, 0
        
        # Object entity/literal statistics
        if node.object.type == "URI":
            obj_freq = self.get_value_dict(self.ent_freq, node.object.node_label)
            obj_subj_freq = self.get_value_dict(self.ent_subj, node.object.node_label)
            obj_obj_freq = self.get_value_dict(self.ent_obj, node.object.node_label)
            
            if self.scaling == "binner":
                obj_freq = self.ent_freq_scaler.transform([[obj_freq]])[0]
                obj_subj_freq = self.ent_subj_scaler.transform([[obj_subj_freq]])[0]
                obj_obj_freq = self.ent_obj_scaler.transform([[obj_obj_freq]])[0]
            
            if self.lit_path is not None:
                lit_freq_o = self._get_lit_scale_no_values()
                lit_type_la = LiteralsFeaturizer.feat(node.object)
        elif node.object.type == "LIT":
            if self.scaling == "binner":
                obj_freq = np.zeros(self.ent_freq_scaler.n_bins_[0])
                obj_subj_freq = np.zeros(self.ent_subj_scaler.n_bins_[0])
                obj_obj_freq = np.zeros(self.ent_obj_scaler.n_bins_[0])
            else:
                obj_freq, obj_subj_freq, obj_obj_freq = 0, 0, 0
            
            if self.lit_path is not None:
                lit_freq_val = self.get_value_dict(self.lit_freq, node.object.node_label)
                if self.scaling == "binner":
                    lit_freq_o = self.lit_freq_scaler.transform([[lit_freq_val]])[0]
                else:
                    lit_freq_o = lit_freq_val
                lit_type_la = LiteralsFeaturizer.feat(node.object)
        else:  # Variable
            if self.scaling == "binner":
                obj_freq = np.zeros(self.ent_freq_scaler.n_bins_[0])
                obj_subj_freq = np.zeros(self.ent_subj_scaler.n_bins_[0])
                obj_obj_freq = np.zeros(self.ent_obj_scaler.n_bins_[0])
            else:
                obj_freq, obj_subj_freq, obj_obj_freq = 0, 0, 0
            
            if self.lit_path is not None:
                lit_freq_o = self._get_lit_scale_no_values()
                lit_type_la = LiteralsFeaturizer.feat(node.object)
        
        # Unscaled statistics are scalars; concatenate requires vectors.
        pred_freq = np.atleast_1d(pred_freq)
        pred_lits = np.atleast_1d(pred_lits)
        pred_ents = np.atleast_1d(pred_ents)
        subj_freq = np.atleast_1d(subj_freq)
        subj_subj_freq = np.atleast_1d(subj_subj_freq)
        sub_obj_freq = np.atleast_1d(sub_obj_freq)
        obj_freq = np.atleast_1d(obj_freq)
        obj_subj_freq = np.atleast_1d(obj_subj_freq)
        obj_obj_freq = np.atleast_1d(obj_obj_freq)
        if self.lit_path is not None:
            lit_freq_o = np.atleast_1d(lit_freq_o)
        # Combine all base features
        if self.lit_path is None:
            stat_vec = np.concatenate([
                pred_freq,
                pred_lits,
                pred_ents,
                subj_freq,
                subj_subj_freq,
                sub_obj_freq,
                obj_freq,
                obj_subj_freq,
                obj_obj_freq,
            ])
        else:
            stat_vec = np.concatenate([
                pred_freq,
                pred_lits,
                pred_ents,
                subj_freq,
                subj_subj_freq,
                sub_obj_freq,
                obj_freq,
                obj_subj_freq,
                obj_obj_freq,
                lit_freq_o,
                lit_type_la,
            ])
        
        return np.concatenate([var_vec, stat_vec])
    
    def _get_lit_scale_no_values(self):
        """Return zero vector for literal features when no value is present."""
        if self.scaling == "binner":
            return np.zeros(self.lit_freq_scaler.n_bins_[0])
        else:
            return np.array([0])
