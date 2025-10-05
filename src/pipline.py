"""
Multimodal Hate Speech Detection and Clustering Pipeline
Author: Senior ML Research Engineer
Description: Production-grade system for detecting hate speech using text + image embeddings,
             with clustering analysis and real-world social media post testing.
"""

import os
import warnings
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import requests
from io import BytesIO

# Deep Learning
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import clip
from torchvision import transforms

# ML & Clustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.cluster import DBSCAN
import xgboost as xgb
import umap

# Visualization
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

# Web Scraping
from bs4 import BeautifulSoup
import re

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Central configuration for the pipeline."""
    
    # Paths
    data_dir: str = "./data"
    cache_dir: str = "./cache"
    models_dir: str = "./models"
    results_dir: str = "./results"
    
    # Model parameters
    text_model: str = "sentence-transformers/all-mpnet-base-v2"
    image_model: str = "clip"  # 'clip' or 'resnet50'
    text_dim: int = 768
    image_dim: int = 512
    fused_dim: int = 1280
    
    # Processing
    batch_size: int = 32
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Dimensionality reduction
    reduction_method: str = "umap"  # 'pca', 'umap', 'tsne'
    reduced_dim: int = 128
    
    # Clustering
    clustering_method: str = "dbscan"
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 5
    
    # Classification
    test_size: float = 0.2
    random_state: int = 42
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for dir_path in [self.data_dir, self.cache_dir, self.models_dir, self.results_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

config = Config()

print(f"🚀 Initialized pipeline with device: {config.device}")
print(f"   GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU Device: {torch.cuda.get_device_name(0)}")


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

class DataPreprocessor:
    """Handles data loading and preprocessing for multimodal hate speech dataset."""
    
    def __init__(self, config: Config):
        self.config = config
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if pd.isna(text):
            return ""
        
        text = str(text)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s,.!?-]', '', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def load_dataset(self, csv_path: str, img_dir: str) -> pd.DataFrame:
        """
        Load the multimodal hate speech dataset.
        
        Args:
            csv_path: Path to CSV file with annotations
            img_dir: Directory containing images
            
        Returns:
            DataFrame with processed data
        """
        print("📂 Loading dataset...")
        df = pd.read_csv(csv_path)
        
        # Clean text
        print("🧹 Cleaning text data...")
        if 'text' in df.columns:
            df['cleaned_text'] = df['text'].apply(self.clean_text)
        elif 'tweet' in df.columns:
            df['cleaned_text'] = df['tweet'].apply(self.clean_text)
        
        # Validate image paths
        print("🖼️  Validating image paths...")
        if 'img' in df.columns:
            df['image_path'] = df['img'].apply(lambda x: os.path.join(img_dir, str(x)))
            df['image_exists'] = df['image_path'].apply(os.path.exists)
        elif 'image' in df.columns:
            df['image_path'] = df['image'].apply(lambda x: os.path.join(img_dir, str(x)))
            df['image_exists'] = df['image_path'].apply(os.path.exists)
        
        # Extract label (adjust column name as needed)
        if 'label' in df.columns:
            df['hate_label'] = df['label']
        elif 'hate' in df.columns:
            df['hate_label'] = df['hate']
        
        print(f"✅ Loaded {len(df)} samples")
        print(f"   - Valid images: {df['image_exists'].sum()}")
        print(f"   - Hate samples: {df['hate_label'].sum()}")
        print(f"   - Non-hate samples: {(~df['hate_label'].astype(bool)).sum()}")
        
        return df


# ============================================================================
# EMBEDDING EXTRACTION
# ============================================================================

class MultimodalEmbedder:
    """Extracts embeddings from text and images using state-of-the-art models."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize text model
        print("🔤 Loading text embedding model...")
        self.text_tokenizer = AutoTokenizer.from_pretrained(config.text_model)
        self.text_model = AutoModel.from_pretrained(config.text_model).to(self.device)
        self.text_model.eval()
        
        # Initialize image model
        print("🖼️  Loading image embedding model...")
        if config.image_model == "clip":
            self.image_model, self.image_preprocess = clip.load("ViT-B/32", device=self.device)
            self.image_model.eval()
        else:
            from torchvision.models import resnet50, ResNet50_Weights
            self.image_model = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.image_model = nn.Sequential(*list(self.image_model.children())[:-1])
            self.image_model = self.image_model.to(self.device)
            self.image_model.eval()
            self.image_preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        print(f"✅ Models loaded on {self.device}")
    
    def mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to get sentence embeddings."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    @torch.no_grad()
    def embed_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            show_progress: Whether to show progress bar
            
        Returns:
            Array of shape (n_samples, text_dim)
        """
        embeddings = []
        
        iterator = tqdm(range(0, len(texts), self.config.batch_size), 
                       desc="Embedding texts", disable=not show_progress)
        
        for i in iterator:
            batch_texts = texts[i:i + self.config.batch_size]
            
            # Tokenize
            encoded = self.text_tokenizer(batch_texts, padding=True, truncation=True, 
                                         max_length=512, return_tensors='pt')
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Generate embeddings
            outputs = self.text_model(**encoded)
            batch_embeddings = self.mean_pooling(outputs, encoded['attention_mask'])
            
            embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    @torch.no_grad()
    def embed_images(self, image_paths: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of images.
        
        Args:
            image_paths: List of image file paths
            show_progress: Whether to show progress bar
            
        Returns:
            Array of shape (n_samples, image_dim)
        """
        embeddings = []
        
        iterator = tqdm(image_paths, desc="Embedding images", disable=not show_progress)
        
        for img_path in iterator:
            try:
                # Load and preprocess image
                image = Image.open(img_path).convert('RGB')
                image_tensor = self.image_preprocess(image).unsqueeze(0).to(self.device)
                
                # Generate embedding
                if self.config.image_model == "clip":
                    features = self.image_model.encode_image(image_tensor)
                else:
                    features = self.image_model(image_tensor)
                
                features = features.squeeze().cpu().numpy()
                
                # Normalize if CLIP
                if self.config.image_model == "clip":
                    features = features / np.linalg.norm(features)
                
                embeddings.append(features)
                
            except Exception as e:
                print(f"⚠️  Error processing {img_path}: {e}")
                # Use zero vector for failed images
                embeddings.append(np.zeros(self.config.image_dim))
        
        return np.vstack(embeddings)
    
    def embed_single_text(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        return self.embed_texts([text], show_progress=False)[0]
    
    def embed_single_image(self, image_path: str) -> np.ndarray:
        """Embed a single image."""
        return self.embed_images([image_path], show_progress=False)[0]


# ============================================================================
# FUSION PROCESSOR
# ============================================================================

class FusionProcessor:
    """Handles multimodal fusion and feature engineering."""
    
    def __init__(self, config: Config):
        self.config = config
        self.scaler = StandardScaler()
        self.fitted = False
    
    def fuse_embeddings(self, text_emb: np.ndarray, image_emb: np.ndarray, 
                       meta_features: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fuse text and image embeddings with optional meta-features.
        
        Args:
            text_emb: Text embeddings (n_samples, text_dim)
            image_emb: Image embeddings (n_samples, image_dim)
            meta_features: Optional meta features (n_samples, n_meta)
            
        Returns:
            Fused embeddings (n_samples, fused_dim + n_meta)
        """
        # Concatenate text and image
        fused = np.concatenate([text_emb, image_emb], axis=1)
        
        # Add meta features if provided
        if meta_features is not None:
            if not self.fitted:
                meta_features = self.scaler.fit_transform(meta_features)
                self.fitted = True
            else:
                meta_features = self.scaler.transform(meta_features)
            
            fused = np.concatenate([fused, meta_features], axis=1)
        
        return fused
    
    def extract_meta_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and engineer meta-features from dataframe."""
        meta_features = []
        
        # Text length
        if 'cleaned_text' in df.columns:
            text_len = df['cleaned_text'].str.len().fillna(0).values.reshape(-1, 1)
            meta_features.append(text_len)
        
        # Add more meta features as available in dataset
        # e.g., posting time, user activity, etc.
        
        if meta_features:
            return np.hstack(meta_features)
        
        return None


# ============================================================================
# DIMENSIONALITY REDUCTION
# ============================================================================

class DimensionalityReducer:
    """Reduces high-dimensional embeddings for clustering and visualization."""
    
    def __init__(self, config: Config):
        self.config = config
        self.reducer = None
        self.reducer_2d = None  # For visualization
        
    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit reducer and transform embeddings.
        
        Args:
            embeddings: High-dimensional embeddings
            
        Returns:
            Reduced embeddings
        """
        print(f"🔬 Applying {self.config.reduction_method.upper()} reduction...")
        
        if self.config.reduction_method == "pca":
            self.reducer = PCA(n_components=self.config.reduced_dim, random_state=self.config.random_state)
        elif self.config.reduction_method == "umap":
            self.reducer = umap.UMAP(n_components=self.config.reduced_dim, 
                                    random_state=self.config.random_state, n_jobs=-1)
        else:
            raise ValueError(f"Unknown reduction method: {self.config.reduction_method}")
        
        reduced = self.reducer.fit_transform(embeddings)
        
        # Also create 2D version for visualization
        print("🎨 Creating 2D projection for visualization...")
        if self.config.reduction_method == "pca":
            self.reducer_2d = PCA(n_components=2, random_state=self.config.random_state)
        else:
            self.reducer_2d = umap.UMAP(n_components=2, random_state=self.config.random_state, n_jobs=-1)
        
        return reduced
    
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform new embeddings using fitted reducer."""
        if self.reducer is None:
            raise ValueError("Reducer not fitted. Call fit_transform first.")
        return self.reducer.transform(embeddings)
    
    def get_2d_projection(self, embeddings: np.ndarray) -> np.ndarray:
        """Get 2D projection for visualization."""
        if self.reducer_2d is None:
            raise ValueError("2D reducer not created. Call fit_transform first.")
        return self.reducer_2d.fit_transform(embeddings)


# ============================================================================
# HATE SPEECH CLASSIFIER
# ============================================================================

class HateSpeechClassifier:
    """XGBoost-based classifier for hate speech detection."""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.feature_importance = None
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """
        Train XGBoost classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Training metrics dictionary
        """
        print("🎯 Training XGBoost classifier...")
        
        # Calculate scale_pos_weight for class imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=self.config.random_state,
            tree_method='gpu_hist' if self.config.device == 'cuda' else 'hist',
            gpu_id=0 if self.config.device == 'cuda' else None,
            eval_metric='auc'
        )
        
        # Train with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Get predictions
        y_pred = self.model.predict(X_val)
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        metrics = {
            'classification_report': classification_report(y_val, y_pred),
            'confusion_matrix': confusion_matrix(y_val, y_pred),
            'roc_auc': roc_auc_score(y_val, y_pred_proba)
        }
        
        # Store feature importance
        self.feature_importance = self.model.feature_importances_
        
        print(f"✅ Training complete. ROC-AUC: {metrics['roc_auc']:.4f}")
        print("\nClassification Report:")
        print(metrics['classification_report'])
        
        return metrics
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict hate speech labels.
        
        Args:
            X: Features
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        return predictions, probabilities
    
    def save(self, path: str):
        """Save trained model."""
        if self.model is None:
            raise ValueError("No model to save.")
        self.model.save_model(path)
        print(f"💾 Model saved to {path}")
    
    def load(self, path: str):
        """Load trained model."""
        self.model = xgb.XGBClassifier()
        self.model.load_model(path)
        print(f"📂 Model loaded from {path}")


# ============================================================================
# CLUSTER ANALYZER
# ============================================================================

class ClusterAnalyzer:
    """Performs clustering and coordinated attack detection."""
    
    def __init__(self, config: Config):
        self.config = config
        self.clustering_model = None
        self.cluster_labels = None
        
    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit clustering model and predict cluster labels.
        
        Args:
            embeddings: Reduced embeddings
            
        Returns:
            Cluster labels (-1 for noise)
        """
        print(f"🔍 Performing {self.config.clustering_method.upper()} clustering...")
        
        if self.config.clustering_method == "dbscan":
            self.clustering_model = DBSCAN(
                eps=self.config.dbscan_eps,
                min_samples=self.config.dbscan_min_samples,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown clustering method: {self.config.clustering_method}")
        
        self.cluster_labels = self.clustering_model.fit_predict(embeddings)
        
        n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
        n_noise = list(self.cluster_labels).count(-1)
        
        print(f"✅ Found {n_clusters} clusters and {n_noise} noise points")
        
        return self.cluster_labels
    
    def analyze_clusters(self, df: pd.DataFrame, hate_predictions: np.ndarray) -> pd.DataFrame:
        """
        Analyze clusters for coordinated attacks.
        
        Args:
            df: Original dataframe
            hate_predictions: Hate speech predictions
            
        Returns:
            DataFrame with cluster statistics
        """
        df_analysis = df.copy()
        df_analysis['cluster'] = self.cluster_labels
        df_analysis['hate_pred'] = hate_predictions
        
        cluster_stats = []
        
        for cluster_id in sorted(set(self.cluster_labels)):
            if cluster_id == -1:  # Skip noise
                continue
            
            cluster_data = df_analysis[df_analysis['cluster'] == cluster_id]
            
            stats = {
                'cluster_id': cluster_id,
                'size': len(cluster_data),
                'hate_ratio': cluster_data['hate_pred'].mean(),
                'hate_count': cluster_data['hate_pred'].sum(),
                'suspiciousness_score': len(cluster_data) * cluster_data['hate_pred'].mean()
            }
            
            # Extract users if available
            if 'user' in cluster_data.columns or 'user_id' in cluster_data.columns:
                user_col = 'user' if 'user' in cluster_data.columns else 'user_id'
                stats['unique_users'] = cluster_data[user_col].nunique()
                stats['top_users'] = cluster_data[user_col].value_counts().head(5).to_dict()
            
            cluster_stats.append(stats)
        
        cluster_df = pd.DataFrame(cluster_stats)
        cluster_df = cluster_df.sort_values('suspiciousness_score', ascending=False)
        
        print("\n🚨 Top Suspicious Clusters:")
        print(cluster_df.head(10))
        
        return cluster_df


# ============================================================================
# VISUALIZATION SUITE
# ============================================================================

class Visualizer:
    """Comprehensive visualization suite for analysis."""
    
    def __init__(self, config: Config):
        self.config = config
        self.results_dir = Path(config.results_dir)
        
    def plot_embedding_projection(self, embeddings_2d: np.ndarray, labels: np.ndarray, 
                                  title: str = "Embedding Projection"):
        """Plot 2D embedding projection colored by labels."""
        fig = px.scatter(
            x=embeddings_2d[:, 0], y=embeddings_2d[:, 1],
            color=labels.astype(str),
            title=title,
            labels={'x': 'Dimension 1', 'y': 'Dimension 2', 'color': 'Label'},
            width=1000, height=700
        )
        fig.write_html(self.results_dir / f"{title.replace(' ', '_').lower()}.html")
        fig.show()
        
    def plot_clusters(self, embeddings_2d: np.ndarray, cluster_labels: np.ndarray):
        """Plot cluster assignments."""
        fig = px.scatter(
            x=embeddings_2d[:, 0], y=embeddings_2d[:, 1],
            color=cluster_labels.astype(str),
            title="Cluster Assignments",
            labels={'x': 'Dimension 1', 'y': 'Dimension 2', 'color': 'Cluster'},
            width=1000, height=700
        )
        fig.write_html(self.results_dir / "clusters.html")
        fig.show()
    
    def plot_confusion_matrix(self, cm: np.ndarray):
        """Plot confusion matrix."""
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'confusion_matrix.png', dpi=300)
        plt.show()
    
    def plot_feature_importance(self, importance: np.ndarray, top_n: int = 20):
        """Plot feature importance from XGBoost."""
        indices = np.argsort(importance)[-top_n:]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(top_n), importance[indices])
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([f'Feature {i}' for i in indices])
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'feature_importance.png', dpi=300)
        plt.show()
    
    def plot_cluster_distribution(self, cluster_df: pd.DataFrame):
        """Plot cluster size and hate ratio distributions."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Cluster sizes
        axes[0].bar(cluster_df['cluster_id'], cluster_df['size'])
        axes[0].set_xlabel('Cluster ID')
        axes[0].set_ylabel('Size')
        axes[0].set_title('Cluster Sizes')
        
        # Hate ratios
        axes[1].bar(cluster_df['cluster_id'], cluster_df['hate_ratio'])
        axes[1].set_xlabel('Cluster ID')
        axes[1].set_ylabel('Hate Ratio')
        axes[1].set_title('Cluster Hate Ratios')
        axes[1].axhline(y=0.5, color='r', linestyle='--', label='50% threshold')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'cluster_distributions.png', dpi=300)
        plt.show()


# ============================================================================
# LIVE POST TESTER
# ============================================================================

class LivePostTester:
    """Test the model on real-world social media posts from URLs."""
    
    def __init__(self, embedder: MultimodalEmbedder, classifier: HateSpeechClassifier,
                 reducer: DimensionalityReducer, fusion_processor: FusionProcessor):
        self.embedder = embedder
        self.classifier = classifier
        self.reducer = reducer
        self.fusion_processor = fusion_processor
        
    def parse_instagram_post(self, url: str) -> Dict[str, Union[str, Image.Image]]:
        """
        Parse Instagram post from URL.
        Note: Instagram scraping is complex due to authentication requirements.
        This is a simplified version that may require API access or authentication.
        """
        print(f"🔍 Parsing Instagram post: {url}")
        
        # For production, you would use Instagram's API or authenticated scraping
        # This is a placeholder implementation
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract text (caption) - this varies based on Instagram's HTML structure
            text = ""
            meta_desc = soup.find('meta', property='og:description')
            if meta_desc:
                text = meta_desc.get('content', '')
            
            # Extract image URL
            image_url = None
            meta_image = soup.find('meta', property='og:image')
            if meta_image:
                image_url = meta_image.get('content')
            
            # Download image
            image = None
            if image_url:
                img_response = requests.get(image_url, timeout=10)
                image = Image.open(BytesIO(img_response.content))
            
            return {'text': text, 'image': image, 'url': url}
            
        except Exception as e:
            print(f"⚠️  Error parsing Twitter post: {e}")
            print("💡 Tip: Consider using Twitter API v2 for reliable access")
            return {'text': '', 'image': None, 'url': url}
    
    def test_post(self, url: str, reference_embeddings: Optional[np.ndarray] = None,
                  reference_labels: Optional[np.ndarray] = None) -> Dict:
        """
        Test a social media post for hate speech.
        
        Args:
            url: URL of the social media post
            reference_embeddings: Optional 2D embeddings for visualization
            reference_labels: Optional reference labels for context
            
        Returns:
            Dictionary with prediction results
        """
        print("\n" + "="*70)
        print("🧪 TESTING LIVE POST")
        print("="*70)
        
        # Parse the post
        if 'instagram.com' in url:
            post_data = self.parse_instagram_post(url)
        elif 'twitter.com' in url or 'x.com' in url:
            post_data = self.parse_twitter_post(url)
        else:
            raise ValueError("Unsupported platform. Only Instagram and Twitter/X are supported.")
        
        if not post_data['text'] and post_data['image'] is None:
            print("❌ Failed to extract content from the post")
            return {'error': 'Failed to extract content'}
        
        print(f"\n📝 Extracted text: {post_data['text'][:200]}...")
        print(f"🖼️  Image extracted: {'Yes' if post_data['image'] else 'No'}")
        
        # Generate embeddings
        print("\n🔄 Generating embeddings...")
        
        # Text embedding
        from DataPreprocessor import DataPreprocessor
        preprocessor = DataPreprocessor(config)
        cleaned_text = preprocessor.clean_text(post_data['text'])
        text_embedding = self.embedder.embed_single_text(cleaned_text)
        
        # Image embedding
        if post_data['image']:
            # Save temporarily
            temp_img_path = Path(config.cache_dir) / 'temp_test_image.jpg'
            post_data['image'].save(temp_img_path)
            image_embedding = self.embedder.embed_single_image(str(temp_img_path))
            temp_img_path.unlink()  # Clean up
        else:
            # Use zero vector if no image
            image_embedding = np.zeros(config.image_dim)
            print("⚠️  No image found, using zero vector")
        
        # Fuse embeddings
        fused_embedding = self.fusion_processor.fuse_embeddings(
            text_embedding.reshape(1, -1),
            image_embedding.reshape(1, -1)
        )
        
        # Reduce dimensionality
        reduced_embedding = self.reducer.transform(fused_embedding)
        
        # Predict
        print("\n🎯 Making prediction...")
        prediction, probability = self.classifier.predict(reduced_embedding)
        
        # Prepare results
        results = {
            'url': url,
            'text': post_data['text'],
            'has_image': post_data['image'] is not None,
            'prediction': 'HATE SPEECH' if prediction[0] == 1 else 'NOT HATE SPEECH',
            'confidence': float(probability[0]),
            'hate_probability': float(probability[0]),
            'safe_probability': float(1 - probability[0])
        }
        
        # Display results
        print("\n" + "="*70)
        print("📊 RESULTS")
        print("="*70)
        print(f"🎯 Prediction: {results['prediction']}")
        print(f"📈 Confidence: {results['confidence']:.2%}")
        print(f"🚨 Hate Probability: {results['hate_probability']:.2%}")
        print(f"✅ Safe Probability: {results['safe_probability']:.2%}")
        
        # Find nearest neighbors if reference embeddings provided
        if reference_embeddings is not None:
            print("\n🔍 Finding similar posts in dataset...")
            test_emb_2d = self.reducer.reducer_2d.transform(reduced_embedding)
            
            # Calculate distances
            from sklearn.metrics.pairwise import euclidean_distances
            distances = euclidean_distances(test_emb_2d, reference_embeddings)[0]
            nearest_indices = np.argsort(distances)[:5]
            
            results['nearest_neighbors'] = {
                'indices': nearest_indices.tolist(),
                'distances': distances[nearest_indices].tolist()
            }
            
            print("\n📍 Top 5 Most Similar Posts:")
            for i, idx in enumerate(nearest_indices, 1):
                label = 'Hate' if reference_labels[idx] == 1 else 'Safe'
                print(f"   {i}. Index {idx} (Distance: {distances[idx]:.4f}) - {label}")
        
        print("="*70 + "\n")
        
        return results


# ============================================================================
# MAIN PIPELINE ORCHESTRATOR
# ============================================================================

class MultimodalHateSpeechPipeline:
    """Main pipeline orchestrator for the entire system."""
    
    def __init__(self, config: Config):
        self.config = config
        self.preprocessor = DataPreprocessor(config)
        self.embedder = MultimodalEmbedder(config)
        self.fusion_processor = FusionProcessor(config)
        self.reducer = DimensionalityReducer(config)
        self.classifier = HateSpeechClassifier(config)
        self.cluster_analyzer = ClusterAnalyzer(config)
        self.visualizer = Visualizer(config)
        
        self.df = None
        self.text_embeddings = None
        self.image_embeddings = None
        self.fused_embeddings = None
        self.reduced_embeddings = None
        self.embeddings_2d = None
        self.cluster_labels = None
        self.predictions = None
        
    def load_and_preprocess(self, csv_path: str, img_dir: str):
        """Load and preprocess the dataset."""
        self.df = self.preprocessor.load_dataset(csv_path, img_dir)
        
    def extract_embeddings(self, use_cache: bool = True):
        """Extract text and image embeddings."""
        cache_file = Path(self.config.cache_dir) / 'embeddings.npz'
        
        if use_cache and cache_file.exists():
            print("📦 Loading cached embeddings...")
            cached = np.load(cache_file)
            self.text_embeddings = cached['text']
            self.image_embeddings = cached['image']
            print("✅ Embeddings loaded from cache")
            return
        
        # Extract embeddings
        print("\n🔤 Extracting text embeddings...")
        self.text_embeddings = self.embedder.embed_texts(self.df['cleaned_text'].tolist())
        
        print("\n🖼️  Extracting image embeddings...")
        valid_images = self.df[self.df['image_exists']]['image_path'].tolist()
        self.image_embeddings = self.embedder.embed_images(valid_images)
        
        # Save to cache
        np.savez_compressed(cache_file, 
                          text=self.text_embeddings, 
                          image=self.image_embeddings)
        print(f"💾 Embeddings cached to {cache_file}")
        
    def fuse_and_reduce(self):
        """Fuse modalities and reduce dimensionality."""
        print("\n🔗 Fusing multimodal embeddings...")
        
        # Extract meta features
        meta_features = self.fusion_processor.extract_meta_features(self.df)
        
        # Fuse embeddings
        self.fused_embeddings = self.fusion_processor.fuse_embeddings(
            self.text_embeddings,
            self.image_embeddings,
            meta_features
        )
        
        print(f"   Fused embedding shape: {self.fused_embeddings.shape}")
        
        # Reduce dimensionality
        self.reduced_embeddings = self.reducer.fit_transform(self.fused_embeddings)
        self.embeddings_2d = self.reducer.get_2d_projection(self.fused_embeddings)
        
        print(f"   Reduced embedding shape: {self.reduced_embeddings.shape}")
        
    def train_classifier(self):
        """Train the hate speech classifier."""
        print("\n🎓 Training classifier...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            self.reduced_embeddings,
            self.df['hate_label'].values,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=self.df['hate_label'].values
        )
        
        # Train
        metrics = self.classifier.train(X_train, y_train, X_val, y_val)
        
        # Visualize results
        self.visualizer.plot_confusion_matrix(metrics['confusion_matrix'])
        self.visualizer.plot_feature_importance(self.classifier.feature_importance)
        
        # Save model
        model_path = Path(self.config.models_dir) / 'hate_speech_classifier.json'
        self.classifier.save(str(model_path))
        
        return metrics
        
    def perform_clustering(self):
        """Perform clustering analysis."""
        print("\n🔬 Performing clustering analysis...")
        
        # Get predictions on full dataset
        self.predictions, _ = self.classifier.predict(self.reduced_embeddings)
        
        # Cluster
        self.cluster_labels = self.cluster_analyzer.fit_predict(self.reduced_embeddings)
        
        # Analyze clusters
        cluster_stats = self.cluster_analyzer.analyze_clusters(self.df, self.predictions)
        
        # Save cluster stats
        cluster_stats.to_csv(Path(self.config.results_dir) / 'cluster_statistics.csv', index=False)
        
        # Visualize
        self.visualizer.plot_clusters(self.embeddings_2d, self.cluster_labels)
        self.visualizer.plot_cluster_distribution(cluster_stats)
        
        return cluster_stats
        
    def visualize_embeddings(self):
        """Create embedding visualizations."""
        print("\n🎨 Creating visualizations...")
        
        self.visualizer.plot_embedding_projection(
            self.embeddings_2d,
            self.df['hate_label'].values,
            title="Hate Speech Embedding Projection"
        )
        
    def save_pipeline(self):
        """Save all pipeline components."""
        print("\n💾 Saving pipeline components...")
        
        # Save reducer
        with open(Path(self.config.models_dir) / 'dimensionality_reducer.pkl', 'wb') as f:
            pickle.dump(self.reducer, f)
        
        # Save fusion processor
        with open(Path(self.config.models_dir) / 'fusion_processor.pkl', 'wb') as f:
            pickle.dump(self.fusion_processor, f)
        
        # Save embeddings for testing
        np.savez_compressed(
            Path(self.config.cache_dir) / 'reference_embeddings.npz',
            embeddings_2d=self.embeddings_2d,
            labels=self.df['hate_label'].values
        )
        
        print("✅ Pipeline saved successfully")
        
    def load_pipeline(self):
        """Load saved pipeline components."""
        print("\n📂 Loading pipeline components...")
        
        # Load classifier
        model_path = Path(self.config.models_dir) / 'hate_speech_classifier.json'
        self.classifier.load(str(model_path))
        
        # Load reducer
        with open(Path(self.config.models_dir) / 'dimensionality_reducer.pkl', 'rb') as f:
            self.reducer = pickle.load(f)
        
        # Load fusion processor
        with open(Path(self.config.models_dir) / 'fusion_processor.pkl', 'rb') as f:
            self.fusion_processor = pickle.load(f)
        
        print("✅ Pipeline loaded successfully")
        
    def create_live_tester(self) -> LivePostTester:
        """Create a live post tester instance."""
        return LivePostTester(
            self.embedder,
            self.classifier,
            self.reducer,
            self.fusion_processor
        )
    
    def run_full_pipeline(self, csv_path: str, img_dir: str):
        """Run the complete pipeline from data loading to model training."""
        print("\n" + "="*70)
        print("🚀 STARTING FULL PIPELINE")
        print("="*70 + "\n")
        
        # Step 1: Load and preprocess
        self.load_and_preprocess(csv_path, img_dir)
        
        # Step 2: Extract embeddings
        self.extract_embeddings(use_cache=True)
        
        # Step 3: Fuse and reduce
        self.fuse_and_reduce()
        
        # Step 4: Train classifier
        metrics = self.train_classifier()
        
        # Step 5: Perform clustering
        cluster_stats = self.perform_clustering()
        
        # Step 6: Visualize
        self.visualize_embeddings()
        
        # Step 7: Save everything
        self.save_pipeline()
        
        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETE")
        print("="*70)
        
        return metrics, cluster_stats


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_training():
    """Example: Train the full pipeline on the Kaggle dataset."""
    
    # Initialize pipeline
    pipeline = MultimodalHateSpeechPipeline(config)
    
    # Set your dataset paths
    csv_path = "./data/train.csv"  # Adjust to your dataset path
    img_dir = "./data/img"          # Adjust to your images directory
    
    # Run full pipeline
    metrics, cluster_stats = pipeline.run_full_pipeline(csv_path, img_dir)
    
    return pipeline


def example_testing(pipeline: MultimodalHateSpeechPipeline):
    """Example: Test on a live social media post."""
    
    # Load reference embeddings for similarity comparison
    ref_data = np.load(Path(config.cache_dir) / 'reference_embeddings.npz')
    reference_embeddings = ref_data['embeddings_2d']
    reference_labels = ref_data['labels']
    
    # Create tester
    tester = pipeline.create_live_tester()
    
    # Test examples
    test_urls = [
        "https://twitter.com/user/status/123456789",  # Replace with real URLs
        "https://www.instagram.com/p/ABC123/",
    ]
    
    for url in test_urls:
        results = tester.test_post(url, reference_embeddings, reference_labels)
        
        # Process results as needed
        if 'error' not in results:
            print(f"\nProcessed: {url}")
            print(f"Result: {results['prediction']}")


def example_manual_testing():
    """Example: Test with manual text and image inputs (without URL parsing)."""
    
    # Load the pipeline
    pipeline = MultimodalHateSpeechPipeline(config)
    pipeline.load_pipeline()
    
    # Manual inputs
    test_text = "Your test text here"
    test_image_path = "./test_image.jpg"  # Path to your test image
    
    # Process
    from pathlib import Path
    preprocessor = DataPreprocessor(config)
    cleaned_text = preprocessor.clean_text(test_text)
    
    # Generate embeddings
    text_emb = pipeline.embedder.embed_single_text(cleaned_text)
    image_emb = pipeline.embedder.embed_single_image(test_image_path)
    
    # Fuse and predict
    fused = pipeline.fusion_processor.fuse_embeddings(
        text_emb.reshape(1, -1),
        image_emb.reshape(1, -1)
    )
    reduced = pipeline.reducer.transform(fused)
    prediction, probability = pipeline.classifier.predict(reduced)
    
    print(f"\nPrediction: {'HATE SPEECH' if prediction[0] == 1 else 'NOT HATE SPEECH'}")
    print(f"Confidence: {probability[0]:.2%}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║  MULTIMODAL HATE SPEECH DETECTION & CLUSTERING PIPELINE          ║
    ║  Production-Ready System with Live Testing Capabilities          ║
    ╚═══════════════════════════════════════════════════════════════════╝
    
    Usage Instructions:
    
    1. TRAINING MODE:
       pipeline = example_training()
       
    2. TESTING MODE (with URL):
       example_testing(pipeline)
       
    3. MANUAL TESTING MODE (without URL):
       example_manual_testing()
    
    Make sure to:
    - Update paths in Config class to match your dataset location
    - Install required packages: pip install -r requirements.txt
    - Have CUDA-enabled GPU for optimal performance
    
    """)
    
    # Uncomment to run:
    # pipeline = example_training()
    # example_testing(pipeline) 'image': image, 'url': url}
            
        except Exception as e:
            print(f"⚠️  Error parsing Instagram post: {e}")
            print("💡 Tip: Instagram scraping may require authentication or API access")
            return {'text': '', 'image': None, 'url': url}
    
    def parse_twitter_post(self, url: str) -> Dict[str, Union[str, Image.Image]]:
        """
        Parse Twitter/X post from URL.
        Note: Twitter API v2 is recommended for production use.
        """
        print(f"🔍 Parsing Twitter post: {url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract text
            text = ""
            meta_desc = soup.find('meta', property='og:description')
            if meta_desc:
                text = meta_desc.get('content', '')
            
            # Extract image
            image_url = None
            meta_image = soup.find('meta', property='og:image')
            if meta_image:
                image_url = meta_image.get('content')
            
            # Download image
            image = None
            if image_url:
                img_response = requests.get(image_url, timeout=10)
                image = Image.open(BytesIO(img_response.content))
            
            return {'text': text,