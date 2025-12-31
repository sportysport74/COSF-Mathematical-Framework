"""
Neural Network Pattern Recognition for COSF Convergences
Train ML model to predict and discover hidden patterns in φ/e relationships

Author: Sportysport
Hardware: RTX 5090 + Ryzen 5900X
Date: December 31, 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime


class ConvergenceDataset(Dataset):
    """Dataset of known convergence points"""

    def __init__(self, csv_file):
        """Load convergence data from CSV"""
        self.data = pd.read_csv(csv_file)

        # Features: n, m, phi_n, e_m
        self.features = torch.tensor(
            self.data[['n', 'm', 'phi_n', 'e_m']].values,
            dtype=torch.float32
        )

        # Normalize features (log scale for large values)
        self.features[:, 2] = torch.log10(self.features[:, 2] + 1)  # log(phi_n)
        self.features[:, 3] = torch.log10(self.features[:, 3] + 1)  # log(e_m)

        # Target: deviation (how close to 1.0)
        self.targets = torch.tensor(
            self.data['deviation'].values,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class ConvergencePredictor(nn.Module):
    """
    Deep neural network to predict convergence quality

    Architecture:
    - Input: (n, m, log(φⁿ), log(eᵐ))
    - Hidden layers with residual connections
    - Output: predicted deviation from 1.0
    """

    def __init__(self, hidden_dims=[256, 512, 512, 256, 128]):
        super().__init__()

        # Input layer
        self.input_layer = nn.Linear(4, hidden_dims[0])

        # Hidden layers with batch norm and dropout
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.BatchNorm1d(hidden_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(0.2)
            ))

        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Xavier initialization for better convergence"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """Forward pass"""
        x = torch.relu(self.input_layer(x))

        for layer in self.hidden_layers:
            x = layer(x)

        x = self.output_layer(x)
        return x.squeeze()


class PatternFinder:
    """Train and use neural networks to find convergence patterns"""

    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        self.model = None
        self.train_losses = []
        self.val_losses = []

        print(f"🧠 Pattern Finder initialized on {device}")
        if device == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")

    def train(self, csv_file, epochs=100, batch_size=64,
              learning_rate=0.001, val_split=0.2):
        """
        Train neural network on known convergences

        Args:
            csv_file: Path to convergence CSV
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            val_split: Fraction of data for validation
        """
        print(f"\n{'='*80}")
        print(f"🧠 TRAINING NEURAL PATTERN FINDER")
        print(f"{'='*80}")

        # Load dataset
        dataset = ConvergenceDataset(csv_file)
        print(f"Loaded {len(dataset)} convergence points")

        # Train/val split
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        print(f"Train set: {train_size} | Val set: {val_size}")

        # Initialize model
        self.model = ConvergencePredictor().to(self.device)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )

        # Training loop
        best_val_loss = float('inf')
        print(f"\nTraining for {epochs} epochs...")

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for features, targets in train_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for features, targets in val_loader:
                    features = features.to(self.device)
                    targets = targets.to(self.device)
                    outputs = self.model(features)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            self.val_losses.append(val_loss)

            scheduler.step(val_loss)

            # Progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"LR: {optimizer.param_groups[0]['lr']:.6f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model("best_model.pt")

        print(f"\n✅ Training complete!")
        print(f"   Best validation loss: {best_val_loss:.6f}")

        # Plot training curves
        self.plot_training_curves()

    def predict(self, n_values, m_values):
        """
        Predict convergence quality for given (n, m) pairs

        Args:
            n_values: Array of n values
            m_values: Array of m values

        Returns:
            Predicted deviations
        """
        if self.model is None:
            raise ValueError("Model not trained! Call train() first.")

        self.model.eval()

        # Compute phi_n and e_m
        phi = (1 + np.sqrt(5)) / 2
        phi_n = phi ** n_values
        e_m = np.exp(m_values)

        # Create features
        features = torch.tensor(
            np.column_stack([n_values, m_values,
                           np.log10(phi_n + 1),
                           np.log10(e_m + 1)]),
            dtype=torch.float32,
            device=self.device
        )

        # Predict
        with torch.no_grad():
            predictions = self.model(features)

        return predictions.cpu().numpy()

    def discover_new_patterns(self, n_range=(1, 10000), n_samples=100000):
        """
        Use trained model to discover potential new convergence regions

        Args:
            n_range: Range of n values to explore
            n_samples: Number of random samples to evaluate

        Returns:
            DataFrame of promising candidates
        """
        print(f"\n{'='*80}")
        print(f"🔍 DISCOVERING NEW PATTERNS")
        print(f"{'='*80}")
        print(f"Sampling {n_samples:,} random (n, m) pairs...")

        # Random sampling
        n_values = np.random.randint(n_range[0], n_range[1], n_samples)
        m_values = np.random.uniform(2, 100, n_samples)

        # Predict in batches
        batch_size = 10000
        all_predictions = []

        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch_n = n_values[i:batch_end]
            batch_m = m_values[i:batch_end]

            predictions = self.predict(batch_n, batch_m)
            all_predictions.extend(predictions)

            if (i + batch_size) % 50000 == 0:
                print(f"  Processed {i+batch_size:,}/{n_samples:,} samples...")

        predictions = np.array(all_predictions)

        # Find promising candidates (low predicted deviation)
        threshold = 0.01  # 1%
        promising_mask = predictions < threshold

        print(f"\n✅ Found {promising_mask.sum():,} promising candidates!")

        # Create results DataFrame
        results = pd.DataFrame({
            'n': n_values[promising_mask],
            'm': m_values[promising_mask],
            'predicted_deviation': predictions[promising_mask]
        })

        results = results.sort_values('predicted_deviation')
        return results

    def plot_training_curves(self):
        """Plot training and validation loss curves"""
        output_dir = Path("results/neural_patterns")
        output_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', alpha=0.8)
        plt.plot(self.val_losses, label='Validation Loss', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Neural Pattern Finder - Training Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(output_dir / f"training_curves_{timestamp}.png", dpi=150)
        print(f"\n📊 Training curves saved to results/neural_patterns/")

    def save_model(self, filename):
        """Save trained model"""
        output_dir = Path("results/neural_patterns")
        output_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), output_dir / filename)

    def load_model(self, filename):
        """Load trained model"""
        model_path = Path("results/neural_patterns") / filename
        self.model = ConvergencePredictor().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        print(f"✅ Model loaded from {model_path}")


def main():
    """Main execution"""

    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                                                                          ║
    ║          🧠 NEURAL PATTERN FINDER FOR COSF FRAMEWORK 🧠                 ║
    ║                                                                          ║
    ║          Using deep learning to discover hidden patterns                ║
    ║          Powered by: RTX 5090 + PyTorch                                ║
    ║                                                                          ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)

    # Initialize pattern finder
    finder = PatternFinder(device='cuda')

    # Find latest GPU search results
    gpu_results = list(Path("results/gpu_search").glob("*.csv"))
    if len(gpu_results) == 0:
        print("⚠️  No GPU search results found!")
        print("   Run cuda_convergence_search.py first to generate training data.")
        return

    latest_results = max(gpu_results, key=lambda p: p.stat().st_mtime)
    print(f"📂 Using training data: {latest_results.name}")

    # Train model
    finder.train(
        csv_file=latest_results,
        epochs=100,
        batch_size=64,
        learning_rate=0.001
    )

    # Discover new patterns
    new_candidates = finder.discover_new_patterns(
        n_range=(1, 10000),
        n_samples=100000
    )

    # Save results
    output_dir = Path("results/neural_patterns")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"neural_discoveries_{timestamp}.csv"

    new_candidates.to_csv(output_file, index=False)
    print(f"\n💾 Discoveries saved to: {output_file}")

    # Display top findings
    print(f"\n{'='*80}")
    print(f"TOP 20 NEURAL DISCOVERIES")
    print(f"{'='*80}")
    print(new_candidates.head(20).to_string(index=False))

    print(f"\n{'='*80}")
    print(f"🧠 NEURAL PATTERN FINDING COMPLETE! 🧠")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
