"""
Publication-Quality Toroidal Geometry Visualization

Creates stunning 3D renderings of nested φ-scaled toroidal shells
with the Flower of Life pattern. Designed for academic publications.

Requires: plotly, numpy, PIL (optional for export)
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_torus_surface(R, r, resolution=100):
    """Generate high-resolution torus surface"""
    u = np.linspace(0, 2*np.pi, resolution)
    v = np.linspace(0, 2*np.pi, resolution)
    u, v = np.meshgrid(u, v)
    
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    
    return x, y, z

def create_publication_visualization():
    """Create multi-panel publication figure"""
    
    phi = (1 + np.sqrt(5)) / 2
    
    # Create 2x2 subplot
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}],
               [{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=(
            'Single Torus (R, r)',
            '3 Nested Shells (φⁿ scaling)',
            '7 Matryoshka Shells',
            'Cross-Section View'
        ),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # Panel 1: Single torus
    R0, r0 = 10, 2
    x, y, z = create_torus_surface(R0, r0)
    
    fig.add_trace(
        go.Surface(x=x, y=y, z=z, 
                  colorscale='Viridis',
                  showscale=False,
                  opacity=0.9),
        row=1, col=1
    )
    
    # Panel 2: 3 nested shells
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for i in range(3):
        Rn = R0 * (phi ** (-i))
        rn = r0 * (phi ** (-i))
        x, y, z = create_torus_surface(Rn, rn, resolution=80)
        
        fig.add_trace(
            go.Surface(x=x, y=y, z=z,
                      colorscale=[[0, colors[i]], [1, colors[i]]],
                      showscale=False,
                      opacity=0.7),
            row=1, col=2
        )
    
    # Panel 3: 7 shells (full Matryoshka)
    colorscale_full = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', 
                       '#98D8C8', '#F7DC6F', '#BB8FCE']
    
    for i in range(7):
        Rn = R0 * (phi ** (-i))
        rn = r0 * (phi ** (-i))
        x, y, z = create_torus_surface(Rn, rn, resolution=60)
        
        fig.add_trace(
            go.Surface(x=x, y=y, z=z,
                      colorscale=[[0, colorscale_full[i]], [1, colorscale_full[i]]],
                      showscale=False,
                      opacity=0.6),
            row=2, col=1
        )
    
    # Panel 4: Cross-section with circles
    theta = np.linspace(0, 2*np.pi, 100)
    
    # Create pseudo-3D cross-section
    for i in range(5):
        Rn = R0 * (phi ** (-i))
        rn = r0 * (phi ** (-i))
        
        # Outer circle
        x_outer = (Rn + rn) * np.cos(theta)
        y_outer = (Rn + rn) * np.sin(theta)
        z_outer = np.zeros_like(theta)
        
        # Inner circle  
        x_inner = (Rn - rn) * np.cos(theta)
        y_inner = (Rn - rn) * np.sin(theta)
        z_inner = np.zeros_like(theta)
        
        color = colorscale_full[i]
        
        fig.add_trace(
            go.Scatter3d(x=x_outer, y=y_outer, z=z_outer,
                        mode='lines',
                        line=dict(color=color, width=4),
                        showlegend=False),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter3d(x=x_inner, y=y_inner, z=z_inner,
                        mode='lines',
                        line=dict(color=color, width=4),
                        showlegend=False),
            row=2, col=2
        )
    
    # Update layout
    camera = dict(eye=dict(x=1.5, y=1.5, z=1.2))
    
    fig.update_scenes(
        camera=camera,
        aspectmode='data',
        xaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
        zaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False)
    )
    
    fig.update_layout(
        title=dict(
            text='COSF Framework: Nested Toroidal Shells with φ Scaling',
            x=0.5,
            xanchor='center',
            font=dict(size=20, family='Arial Black')
        ),
        width=1600,
        height=1200,
        showlegend=False,
        paper_bgcolor='white',
        font=dict(family='Arial')
    )
    
    # Save
    fig.write_html('../../images/toroidal_geometry/publication_figure.html')
    print(" Saved: images/toroidal_geometry/publication_figure.html")
    
    # Also save as static image if kaleido installed
    try:
        fig.write_image('../../images/toroidal_geometry/publication_figure.png', 
                       width=1600, height=1200, scale=2)
        print(" Saved: images/toroidal_geometry/publication_figure.png (high-res)")
    except:
        print(" Install kaleido for PNG export: pip install kaleido")
    
    fig.show()

if __name__ == "__main__":
    print("Generating publication-quality toroidal visualization...")
    create_publication_visualization()
    print("\n Visualization complete!")
