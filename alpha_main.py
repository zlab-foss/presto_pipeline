from pathlib import Path
import numpy as np
import torch
from typing import Dict, Optional, Tuple

from torch.utils.data import TensorDataset, DataLoader

from main import create_lulc_mask, get_tile_statistics
from data_sources.pysatellite import AlphaEmbeddingDownloader
from utils.utils import  save_pred_to_tiff
import rasterio

import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm import tqdm

from shap_tiler import ShapefileTiler
def initialize_tiler(configs: Dict) -> ShapefileTiler:
    """Initialize shapefile tiler."""
    try:
        tiler = ShapefileTiler(
            shp_path=configs["asset_path"],
            max_pixels=configs["tile_size"],
            temp_dir="./TEMP2/",
        )
        print(f"✅ Initialized tiler for: {configs['asset_path']}")
        return tiler
    except Exception as e:
        raise RuntimeError(f"Failed to initialize tiler: {e}")



        

def read_embeddings_drop_filled_mask(embedding_path, drop_last_band=True):
    """
    Reads embedding GeoTIFF as (H*W, C) float32 features and returns (X, (H,W), meta).
    Drops the last band (FILLED_MASK) if requested.
    """
    with rasterio.open(embedding_path) as src:
        H, W = src.height, src.width
        C = src.count

        # band descriptions may be None
        desc = list(src.descriptions) if src.descriptions else [None] * C

        arr = src.read()  # (C, H, W)
        meta = src.meta.copy()

    if drop_last_band:
        # Prefer dropping last band if it's named FILLED_MASK, otherwise still drop last band
        last_name = (desc[-1] or "").strip() if desc else ""
        if last_name.upper() == "FILLED_MASK" or C == 65:
            arr = arr[:-1, :, :]  # drop last
            C = arr.shape[0]
        # else: keep as-is

    # reshape to (H*W, C)
    X = np.moveaxis(arr, 0, -1).reshape(-1, C).astype(np.float32, copy=False)
    return X, (H, W), meta


def make_loader_from_X(X, batch_size=2048, num_workers=0, pin_memory=False):
    """Create a DataLoader from numpy array X."""
    ds = TensorDataset(torch.from_numpy(X))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


class MLPClassifier(BaseEstimator, ClassifierMixin):
    """Multi-layer perceptron classifier with PyTorch backend."""
    
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', 
                 solver='adam', alpha=0.0001, learning_rate=0.001, 
                 max_iter=200, random_state=None, verbose=True, 
                 early_stopping=False, n_iter_no_change=10, dropout_rate=0.0, device='cpu',
                 class_weight=None, n_classes=None, class_names=None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.device = device
        self.class_weight = class_weight
        self.dropout_rate = dropout_rate
        
        self.model_ = None
        self.optimizer_ = None
        self.n_features_in_ = None
        self.n_classes_ = n_classes
        self.classes_ = class_names
        self.class_weights_ = None
        self.loss_curve_ = []
        self.validation_scores_ = []
        
    def _build_model(self):
        """Build the neural network architecture."""
        layers = []
        input_size = self.n_features_in_

        # Hidden layers
        for hidden_size in self.hidden_layer_sizes:
            layers.append(nn.Linear(input_size, hidden_size))

            # Activation selection
            if self.activation == 'relu':
                layers.append(nn.ReLU())
            elif self.activation == 'tanh':
                layers.append(nn.Tanh())
            elif self.activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif self.activation == 'gelu':
                layers.append(nn.GELU())
            else:
                raise ValueError(f"Unknown activation: {self.activation}")

            # Optional Dropout
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(p=self.dropout_rate))

            input_size = hidden_size

        # Output layer
        layers.append(nn.Linear(input_size, self.n_classes_))

        return nn.Sequential(*layers)
    
    def _get_optimizer(self):
        """Get optimizer (placeholder for compatibility)."""
        if self.solver == 'adam':
            return torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unknown solver: {self.solver}")
  
    def predict(self, data_loader):
        """
        Predict class labels using PyTorch DataLoader.
        
        Parameters
        ----------
        data_loader : DataLoader
            Data as PyTorch DataLoader.
            
        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
            Predicted class labels.
        """
        if self.model_ is None:
            raise ValueError("Model has not been fitted yet.")
        
        if not isinstance(data_loader, DataLoader):
            raise TypeError("data_loader must be a PyTorch DataLoader")
        
        self.model_.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Processing Batches..."):
                # Handle both single tensor and tuple unpacking
                if isinstance(batch, (list, tuple)):
                    batch_X = batch[0]
                else:
                    batch_X = batch
                    
                batch_X = batch_X.to(self.device)
                outputs = self.model_(batch_X)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
        
        return np.array(predictions)
    
    @classmethod
    def load(cls, filepath, device='cpu'):
        """
        Load a saved model from a file.
        
        Parameters
        ----------
        filepath : str
            Path to the saved checkpoint file.
            
        Returns
        -------
        model : MLPClassifier
            The loaded MLPClassifier instance.
        """
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        
        # Handle dropout_rate (default to 0.0 if not present for backward compatibility)
        dropout_rate = checkpoint.get('dropout_rate', 0.0)
        
        model = cls(
            hidden_layer_sizes=checkpoint['hidden_layer_sizes'],
            activation=checkpoint['activation'],
            solver=checkpoint['solver'],
            alpha=checkpoint['alpha'],
            learning_rate=checkpoint['learning_rate'],
            max_iter=checkpoint['max_iter'],
            random_state=checkpoint['random_state'],
            verbose=checkpoint['verbose'],
            early_stopping=checkpoint['early_stopping'],
            n_iter_no_change=checkpoint['n_iter_no_change'],
            dropout_rate=dropout_rate,
            device=device,
            class_weight=checkpoint['class_weight'],
            n_classes=checkpoint['n_classes_'],
        )
        
        model.n_features_in_ = checkpoint['n_features_in_']
        model.n_classes_ = checkpoint['n_classes_']
        
        model.model_ = model._build_model().to(model.device)
        model.model_.load_state_dict(checkpoint['model_state_dict'])
        
        model.optimizer_ = model._get_optimizer()
        if checkpoint['optimizer_state_dict'] is not None:
            model.optimizer_.load_state_dict(checkpoint['optimizer_state_dict'])
            
        if checkpoint['classes_'] is not None:
            model.classes_ = checkpoint['classes_']
        
        model.class_weights_ = checkpoint['class_weights_']
        if torch.is_tensor(model.class_weights_):
            model.class_weights_ = model.class_weights_.to(model.device)
        
        model.loss_curve_ = checkpoint['loss_curve_']
        model.validation_scores_ = checkpoint['validation_scores_']
        
        return model


def validate_config(configs: Dict) -> None:
    """Validate configuration parameters."""
    required_keys = [
        "asset_path",
        "credentials_path",
        "service_account",
        "year",
        "out_dir",
        "tile_size",
    ]

    for key in required_keys:
        if key not in configs:
            raise ValueError(f"Missing required config key: {key}")

    if configs["year"] < 2017:
        raise ValueError(f"Year {configs['year']} is too early for Alpha earth embedding")

    if configs["year"] > 2024:
        raise ValueError(f"Alpha earth not available after 2024. Got year={configs['year']}")


def setup_directories(out_dir: Path) -> None:
    """Create output directory structure."""
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Output directory ready: {out_dir}")
    except Exception as e:
        raise RuntimeError(f"Failed to create output directory: {e}")


def initialize_downloaders(configs: Dict, out_root: Path) -> AlphaEmbeddingDownloader:
    """Initialize appropriate downloaders based on sensor type."""
    try:
        dl_alpha = AlphaEmbeddingDownloader(
            credentials_path=configs["credentials_path"],
            service_account=configs["service_account"],
            output_dir=out_root / "alpha",
        )

        print("✅ Initialized AlphaEmbeddingDownloader")
        return dl_alpha

    except Exception as e:
        raise RuntimeError(f"Failed to initialize downloaders: {e}")



def load_model(configs: Dict) -> MLPClassifier:
    """Load the appropriate model based on configuration."""
    try:
        want_cuda = configs.get("device", "cpu") == "cuda"
        device = torch.device("cuda" if (want_cuda and torch.cuda.is_available()) else "cpu")

        model_path = "./weights/irrigation/Alpha.pth"

        clf = MLPClassifier.load(model_path, device=device)
        print(f"✅ Loaded model from: {model_path}")
        print(f"🖥️  Using device: {device}")

        return clf

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model file not found: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")



def build_masked_dataloader(
    X: np.ndarray,
    lulc_mask: np.ndarray,
    batch_size: int = 2048,
) -> Tuple[Optional[DataLoader], np.ndarray]:
    """
    Restrict the data to only pixels where lulc_mask == 1.

    Parameters
    ----------
    X : np.ndarray
        Feature array of shape (N, C) where N = H*W
    lulc_mask : np.ndarray
        Binary mask of shape (H, W)
    batch_size : int
        Batch size for DataLoader

    Returns
    -------
    new_loader : DataLoader or None
        DataLoader with masked data
    valid_idx : np.ndarray
        Flat indices into the full H*W raster
    """
    flat_mask = lulc_mask.ravel().astype(bool)
    valid_idx = np.where(flat_mask)[0]
    num_valid = int(valid_idx.size)

    if num_valid == 0:
        return None, valid_idx

    # Select only valid pixels
    X_masked = X[valid_idx]
    
    # Create new DataLoader
    new_loader = make_loader_from_X(X_masked, batch_size=batch_size)

    return new_loader, valid_idx


def process_tile(
    item: Dict,
    configs: Dict,
    dl_alpha: AlphaEmbeddingDownloader,
    clf: MLPClassifier,
    out_root: Path,
) -> bool:
    """Process a single tile: download (if needed), load embeddings, predict, save."""
    poly_idx = item["poly_idx"]
    tile_idx = item["tile_idx"]
    year = configs["year"]

    if tile_idx is None:
        tile_tag = "full"
    else:
        ty, tx = tile_idx
        tile_tag = f"t{ty:03d}_{tx:03d}"

    out_name = f"{year}_p{poly_idx:03d}_{tile_tag}.tif"

    try:
        print(f"\n{'=' * 60}")
        print(f"Processing: {out_name}")
        print(f"{'=' * 60}")

        # ===== 1. Resolve expected paths =====
        path_embedding = (out_root / "alpha" / out_name).resolve()

        # ===== 1b. Download embedding (skip if exists) =====
        skip_download = bool(configs.get("skip_download", False))

        if skip_download and path_embedding.exists():
            print(f"♻️  Embedding exists, skipping download: {path_embedding}")
        else:
            print("📥 Downloading Alpha Earth embedding...")
            dl_alpha.download_from_shapefile(
                shp_path=item["shp_path"],
                out_tif=out_name,
                year=year,
            )
            if not path_embedding.exists():
                raise RuntimeError(f"Embedding download finished but file not found: {path_embedding}")

        # ===== 2. Load embeddings =====
        print("📖 Reading Alpha Earth embeddings...")
        X, data_shape, meta = read_embeddings_drop_filled_mask(path_embedding, drop_last_band=True)
        
        H, W = data_shape
        # N = H * W
        print(f"📐 Data shape: {data_shape}, Features: {X.shape[1]}")

        # ===== 3. Create LULC mask (with caching / skip-if-exists) =====
        lulc_mask = create_lulc_mask(
            configs, item, out_name, out_root, data_shape, path_embedding
        )

        if lulc_mask is not None and lulc_mask.shape != (H, W):
            raise RuntimeError(
                f"LULC mask shape {lulc_mask.shape} does not match data shape {(H, W)}"
            )

        # ===== 4. Apply LULC mask BEFORE inference =====
        full_pred = np.zeros((H * W,), dtype=np.uint8)  # default: 0 (nodata / non-crop)

        if lulc_mask is not None:
            flat_mask = lulc_mask.ravel().astype(bool)
            num_valid = int(flat_mask.sum())

            if num_valid == 0:
                print(
                    "⚠️ No valid pixels in this tile after LULC filtering. "
                    "Skipping irrigation inference and saving all-zero raster."
                )

                full_pred = full_pred.reshape(H, W)
                output_dir = out_root / "predictions"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / out_name

                save_pred_to_tiff(
                    out_arr=full_pred,
                    out_tiff=str(output_path),
                    ref_tiff=str(path_embedding),
                    dtype="uint8",
                    nodata=0,
                    compress="lzw",
                )

                print(f"✅ Saved empty (all-zero) prediction: {output_path}")
                return True

            print(f"🔍 Filtering to {num_valid} valid pixels for irrigation inference")
            data_loader, valid_idx = build_masked_dataloader(
                X, lulc_mask, batch_size=configs.get("batch_size", 2048)
            )
        else:
            print("ℹ️ No LULC mask in use; running inference on all pixels")
            valid_idx = np.arange(H * W, dtype=np.int64)
            data_loader = make_loader_from_X(X, batch_size=configs.get("batch_size", 2048))

        if data_loader is None or len(valid_idx) == 0:
            print("⚠️ No valid pixels after masking. Saving all-zero raster and skipping inference.")
            full_pred = full_pred.reshape(H, W)
            output_dir = out_root / "predictions"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / out_name

            save_pred_to_tiff(
                out_arr=full_pred,
                out_tiff=str(output_path),
                ref_tiff=str(path_embedding),
                dtype="uint8",
                nodata=0,
                compress="lzw",
            )
            print(f"✅ Saved empty (all-zero) prediction: {output_path}")
            return True

        # ===== 5. Run prediction on VALID pixels only =====
        print("🤖 Running irrigation classification...")
        pred = clf.predict(data_loader)

        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        else:
            pred = np.asarray(pred)

        pred = pred + 1  # 0->1, 1->2, ...

        if pred.shape[0] != valid_idx.shape[0]:
            raise RuntimeError(
                f"Prediction length {pred.shape[0]} does not match "
                f"number of valid pixels {valid_idx.shape[0]}"
            )

        full_pred[valid_idx] = pred.astype(np.uint8)
        full_pred = full_pred.reshape(H, W)

        # ===== 6. Save result =====
        output_dir = out_root / "predictions"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / out_name

        print("💾 Saving prediction...")
        save_pred_to_tiff(
            out_arr=full_pred,
            out_tiff=str(output_path),
            ref_tiff=str(path_embedding),
            dtype="uint8",
            nodata=0,
            compress="lzw",
        )

        print(f"✅ Successfully saved: {output_path}")
        return True

    except Exception as e:
        print(f"❌ Error processing {out_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_pipeline_alpha(configs: Dict) -> None:
    """Main pipeline execution function."""
    print("\n" + "=" * 60)
    print("🚀 Starting Alpha Earth Irrigation Classification Pipeline")
    print("=" * 60 + "\n")

    try:
        print("🔍 Validating configuration...")
        validate_config(configs)

        out_root = Path(configs["out_dir"])
        setup_directories(out_root)

        dl_alpha = initialize_downloaders(configs, out_root)
        
        tiler = initialize_tiler(configs)

        tile_stats = get_tile_statistics(tiler)
        estimated_total = sum(t["total"] for t in tile_stats["tiles"])

        print("\n📊 Tile Statistics:")
        print(f"   • Number of polygons: {tile_stats['num_polygons']}")
        print(f"   • Estimated total tiles: {estimated_total}")
        for t in tile_stats["tiles"]:
            if t["total"] > 1:
                print(
                    f"   • Polygon {t['poly_idx']}: {t['nx']}×{t['ny']} = "
                    f"{t['total']} tiles ({t['width_m']/1000:.1f}×{t['height_m']/1000:.1f} km)"
                )

        clf = load_model(configs)

        print("\n📊 Counting total tiles...")
        tile_list = list(tiler)
        total_tiles = len(tile_list)
        print(f"✅ Found {total_tiles} tiles to process\n")

        print("\n🔄 Starting tile processing...\n")
        success_count = 0
        fail_count = 0

        for idx, item in enumerate(tile_list, 1):
            if idx < configs.get("tile_idx_resume", -1):
                success_count += 1
                continue
            else:
                print(f"\n📍 Tile {idx}/{total_tiles}")

                success = process_tile(
                    item, configs, dl_alpha, clf, out_root
                )

                if success:
                    success_count += 1
                else:
                    fail_count += 1

        print("\n" + "=" * 60)
        print("🏁 Pipeline Complete!")
        print("=" * 60)
        print(f"✅ Successful tiles: {success_count}")
        print(f"❌ Failed tiles: {fail_count}")
        print(f"📁 Output directory: {out_root / 'predictions'}")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        raise
        

# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    configs = {
        "asset_path": "./ROI/test/patches_season_97_98.shp",
        "credentials_path": "./credentials/earthengine_credentials.json",
        "service_account": "fanapanomaly@fanapanomaly.iam.gserviceaccount.com",
        "year": 2018,
        "out_dir": "./data/test_results/2018/alpha",
        "tile_size": 2024,
        "landuse_method": "ESRI",  # "ESRI", "presto", or "skip"
        "device": "cuda",  # "cuda" or "cpu"
        "tile_idx_resume": -1,
        "skip_download": True,  # if True: skip any download when the expected tif already exists
    }

    try:
        run_pipeline_alpha(configs)
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        exit(1)