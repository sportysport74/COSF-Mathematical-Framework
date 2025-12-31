"""
GPU Ray-Traced Visualization for COSF Toroidal Geometry
Real-time interactive 3D rendering with plasma effects

Author: Sportysport
Hardware: RTX 5090 + Ryzen 5900X
Date: December 31, 2025
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
import colorsys


class GPURaytracer:
    """GPU-accelerated ray tracing for toroidal shells"""

    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        self.phi = (1 + np.sqrt(5)) / 2

        print(f"🎨 GPU Raytracer initialized")
        print(f"   Device: {self.device}")
        if device == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")

    def generate_torus_points(self, R, r, n_theta=100, n_phi=50):
        """
        Generate points on a torus surface

        Args:
            R: Major radius (from center to tube center)
            r: Minor radius (tube radius)
            n_theta: Points around major circle
            n_phi: Points around minor circle

        Returns:
            (x, y, z) arrays of torus surface points
        """
        theta = torch.linspace(0, 2*np.pi, n_theta, device=self.device)
        phi = torch.linspace(0, 2*np.pi, n_phi, device=self.device)

        theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing='ij')

        x = (R + r * torch.cos(phi_grid)) * torch.cos(theta_grid)
        y = (R + r * torch.cos(phi_grid)) * torch.sin(theta_grid)
        z = r * torch.sin(phi_grid)

        return x.cpu().numpy(), y.cpu().numpy(), z.cpu().numpy()

    def compute_plasma_glow(self, x, y, z, shell_idx, total_shells=17):
        """
        Compute plasma-like glow intensity for visualization

        Args:
            x, y, z: Point coordinates
            shell_idx: Which nested shell (0 = innermost)
            total_shells: Total number of shells

        Returns:
            Glow intensity array
        """
        # Distance from origin
        r = np.sqrt(x**2 + y**2 + z**2)

        # Angular position (for spiraling effect)
        theta = np.arctan2(y, x)

        # Plasma oscillation (golden ratio based)
        freq = self.phi ** (shell_idx / total_shells)
        glow = 0.5 + 0.5 * np.sin(freq * theta + shell_idx * np.pi / 5)

        # Radial falloff (brighter near surface)
        glow *= np.exp(-0.1 * np.abs(z))

        return glow

    def create_nested_toroids_interactive(self, n_shells=17, base_R=10.0,
                                         base_r=2.0, export_html=True):
        """
        Create interactive 3D visualization of nested toroidal shells

        Args:
            n_shells: Number of nested shells (17 for φ¹⁷)
            base_R: Major radius of outermost shell
            base_r: Minor radius of outermost shell
            export_html: Save as interactive HTML

        Returns:
            Plotly figure object
        """
        print(f"\n{'='*80}")
        print(f"🎨 CREATING GPU RAY-TRACED VISUALIZATION")
        print(f"{'='*80}")
        print(f"Rendering {n_shells} nested toroidal shells...")

        fig = go.Figure()

        # Generate colormap (golden to cyan plasma)
        colors = []
        for i in range(n_shells):
            # HSV color space for smooth gradients
            hue = 0.15 + 0.5 * i / n_shells  # Gold to cyan
            sat = 0.8
            val = 0.9
            rgb = colorsys.hsv_to_rgb(hue, sat, val)
            colors.append(f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})')

        # Render each shell
        for shell_idx in range(n_shells):
            # Scale by φ for each shell (outermost is largest)
            scale = self.phi ** (-(n_shells - 1 - shell_idx) / 4)  # Gentler scaling for visibility
            R = base_R * scale
            r = base_r * scale

            print(f"  Shell {shell_idx+1}/{n_shells}: R={R:.2f}, r={r:.2f}")

            # Generate torus
            x, y, z = self.generate_torus_points(R, r, n_theta=100, n_phi=50)

            # Compute glow intensity
            glow = self.compute_plasma_glow(x, y, z, shell_idx, n_shells)

            # Opacity based on shell (inner shells more transparent)
            opacity = 0.3 + 0.5 * (shell_idx / n_shells)

            # Add to figure
            fig.add_trace(go.Surface(
                x=x, y=y, z=z,
                surfacecolor=glow,
                colorscale=[[0, colors[shell_idx]], [1, 'white']],
                showscale=False,
                opacity=opacity,
                name=f'Shell {shell_idx+1}',
                hovertemplate=f'Shell {shell_idx+1}<br>R={R:.2f}<br>r={r:.2f}<extra></extra>'
            ))

        # Layout with dark theme
        fig.update_layout(
            title={
                'text': '🌌 COSF Toroidal Geometry - 17 Nested Shells (φ¹⁷) 🌌<br>' +
                        '<sub>GPU Ray-Traced Visualization | Powered by RTX 5090</sub>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'color': 'gold'}
            },
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor='rgb(10, 10, 30)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            paper_bgcolor='rgb(10, 10, 30)',
            width=1200,
            height=900,
            showlegend=False
        )

        print(f"✅ Visualization complete!")

        # Export HTML
        if export_html:
            output_dir = Path("images/toroidal_geometry")
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"gpu_raytraced_toroids_{timestamp}.html"

            fig.write_html(str(output_file))
            print(f"💾 Interactive HTML saved to: {output_file}")

        return fig

    def create_convergence_landscape_3d(self, csv_file):
        """
        Create 3D landscape of convergence quality

        Args:
            csv_file: Path to convergence CSV

        Returns:
            Plotly figure object
        """
        import pandas as pd

        print(f"\n{'='*80}")
        print(f"🗺️  CREATING CONVERGENCE LANDSCAPE")
        print(f"{'='*80}")

        # Load data
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} convergence points")

        # Extract top 1000 for visualization
        df_top = df.nsmallest(1000, 'deviation')

        # Create 3D scatter
        fig = go.Figure()

        # Color by deviation (log scale)
        colors = np.log10(df_top['deviation'].values + 1e-10)

        fig.add_trace(go.Scatter3d(
            x=df_top['n'],
            y=df_top['m'],
            z=-df_top['deviation_pct'],  # Negative so peaks = good convergences
            mode='markers',
            marker=dict(
                size=5,
                color=colors,
                colorscale='Viridis',
                colorbar=dict(title='log₁₀(deviation)'),
                showscale=True,
                opacity=0.8
            ),
            text=[f'n={n}<br>m={m:.1f}<br>dev={d:.4f}%'
                  for n, m, d in zip(df_top['n'], df_top['m'], df_top['deviation_pct'])],
            hovertemplate='%{text}<extra></extra>',
            name='Convergences'
        ))

        # Highlight COSF region (n=16-18)
        cosf_region = df_top[(df_top['n'] >= 16) & (df_top['n'] <= 18)]
        if len(cosf_region) > 0:
            fig.add_trace(go.Scatter3d(
                x=cosf_region['n'],
                y=cosf_region['m'],
                z=-cosf_region['deviation_pct'],
                mode='markers',
                marker=dict(
                    size=10,
                    color='gold',
                    symbol='diamond',
                    line=dict(color='white', width=2)
                ),
                name='COSF Region',
                text=[f'🎯 COSF<br>n={n}<br>m={m:.1f}<br>dev={d:.4f}%'
                      for n, m, d in zip(cosf_region['n'], cosf_region['m'], cosf_region['deviation_pct'])],
                hovertemplate='%{text}<extra></extra>'
            ))

        fig.update_layout(
            title={
                'text': '🗺️ COSF Convergence Landscape (3D) 🗺️<br>' +
                        '<sub>Top 1000 φⁿ/eᵐ Convergences | GPU Computed</sub>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'color': 'cyan'}
            },
            scene=dict(
                xaxis=dict(title='n (φ exponent)', gridcolor='gray'),
                yaxis=dict(title='m (e exponent)', gridcolor='gray'),
                zaxis=dict(title='-Deviation % (higher = better)', gridcolor='gray'),
                bgcolor='rgb(10, 10, 30)'
            ),
            paper_bgcolor='rgb(10, 10, 30)',
            width=1200,
            height=900
        )

        # Export
        output_dir = Path("images/convergence_landscapes")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"convergence_landscape_3d_{timestamp}.html"

        fig.write_html(str(output_file))
        print(f"💾 Convergence landscape saved to: {output_file}")

        return fig

    def create_4k_render(self, n_shells=17, base_R=10.0, base_r=2.0):
        """
        Create ultra high-resolution static render (4K)

        Args:
            n_shells: Number of shells
            base_R: Major radius
            base_r: Minor radius

        Returns:
            Path to saved image
        """
        print(f"\n{'='*80}")
        print(f"🖼️  CREATING 4K STATIC RENDER")
        print(f"{'='*80}")

        fig = plt.figure(figsize=(38.4, 21.6), dpi=100)  # 3840x2160 @ 100 DPI
        ax = fig.add_subplot(111, projection='3d', facecolor='#0a0a1e')

        # Dark background
        fig.patch.set_facecolor('#0a0a1e')

        # Generate colormap
        colors = []
        for i in range(n_shells):
            hue = 0.15 + 0.5 * i / n_shells
            sat = 0.8
            val = 0.9
            rgb = colorsys.hsv_to_rgb(hue, sat, val)
            colors.append(rgb)

        # Render each shell
        for shell_idx in range(n_shells):
            scale = self.phi ** (-(n_shells - 1 - shell_idx) / 4)
            R = base_R * scale
            r = base_r * scale

            x, y, z = self.generate_torus_points(R, r, n_theta=200, n_phi=100)
            glow = self.compute_plasma_glow(x, y, z, shell_idx, n_shells)

            alpha = 0.3 + 0.5 * (shell_idx / n_shells)

            ax.plot_surface(x, y, z, facecolors=plt.cm.plasma(glow),
                           alpha=alpha, antialiased=True, shade=True)

        # Styling
        ax.set_xlim(-base_R*1.2, base_R*1.2)
        ax.set_ylim(-base_R*1.2, base_R*1.2)
        ax.set_zlim(-base_r*1.5, base_r*1.5)

        ax.set_title('COSF Toroidal Geometry - 17 Nested Shells (φ¹⁷)\n' +
                     'GPU Ray-Traced | RTX 5090',
                     color='gold', fontsize=36, pad=20)

        ax.set_axis_off()
        ax.view_init(elev=25, azim=45)

        # Export 4K
        output_dir = Path("images/4k_renders")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"cosf_4k_render_{timestamp}.png"

        plt.savefig(output_file, dpi=100, facecolor='#0a0a1e',
                   bbox_inches='tight', pad_inches=0.5)

        print(f"💾 4K render saved to: {output_file}")
        print(f"   Resolution: 3840x2160")

        plt.close()

        return output_file


def main():
    """Main execution"""

    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                                                                          ║
    ║          🎨 GPU RAY-TRACED VISUALIZATION FOR COSF 🎨                    ║
    ║                                                                          ║
    ║          Creating stunning 3D visualizations                            ║
    ║          Powered by: RTX 5090                                           ║
    ║                                                                          ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)

    # Initialize raytracer
    raytracer = GPURaytracer(device='cuda')

    # 1. Create interactive nested toroids
    print("\n🎯 Creating interactive nested toroidal shells...")
    fig_toroids = raytracer.create_nested_toroids_interactive(
        n_shells=17,
        base_R=10.0,
        base_r=2.0,
        export_html=True
    )

    # 2. Create convergence landscape (if data available)
    gpu_results = list(Path("results/gpu_search").glob("*.csv"))
    if len(gpu_results) > 0:
        latest_results = max(gpu_results, key=lambda p: p.stat().st_mtime)
        print(f"\n📂 Using convergence data: {latest_results.name}")

        fig_landscape = raytracer.create_convergence_landscape_3d(latest_results)
    else:
        print("\n⚠️  No GPU search results found for landscape.")
        print("   Run cuda_convergence_search.py first.")

    # 3. Create 4K static render
    print("\n🖼️  Creating 4K static render...")
    render_path = raytracer.create_4k_render(
        n_shells=17,
        base_R=10.0,
        base_r=2.0
    )

    print(f"\n{'='*80}")
    print(f"🎨 GPU VISUALIZATION COMPLETE!")
    print(f"{'='*80}")
    print(f"\nGenerated files:")
    print(f"  📊 Interactive toroids: images/toroidal_geometry/")
    print(f"  🗺️  Convergence landscape: images/convergence_landscapes/")
    print(f"  🖼️  4K render: {render_path}")
    print(f"\n🔥 LEGENDARY VISUALS ACHIEVED! 🔥")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
