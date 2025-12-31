"""
ULTIMATE COSF Visualization: Glowing Nested Toroidal Shells

This creates the DEFINITIVE publication figure with:
- Glowing plasma effects
- Golden ratio nested shells (7 layers)
- Quantum wave overlay
- Equation annotations
- Professional lighting

Output: 4K resolution, publication-ready
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_glowing_torus(R, r, color, resolution=100, glow_intensity=0.8):
    """Create a torus with glow effect"""
    u = np.linspace(0, 2*np.pi, resolution)
    v = np.linspace(0, 2*np.pi, resolution)
    u, v = np.meshgrid(u, v)
    
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    
    # Add subtle noise for plasma effect
    noise = 0.02 * r * np.random.randn(*x.shape)
    z += noise
    
    return x, y, z

def create_ultimate_visualization():
    """Generate the ultimate COSF visual manifesto"""
    
    phi = (1 + np.sqrt(5)) / 2
    
    # Create figure
    fig = go.Figure()
    
    # Base parameters
    R0, r0 = 10, 2
    
    # Color palette - plasma/energy gradient
    colors = [
        '#0066FF',  # Deep blue
        '#00FFFF',  # Cyan
        '#4ECDC4',  # Teal
        '#45B7D1',  # Light blue
        '#FFD700',  # Gold
        '#FFA500',  # Orange
        '#FF6B6B'   # Coral
    ]
    
    print("Generating 7 nested φ-scaled toroidal shells...")
    
    # Generate 7 shells
    for i in range(7):
        Rn = R0 * (phi ** (-i))
        rn = r0 * (phi ** (-i))
        
        x, y, z = create_glowing_torus(Rn, rn, colors[i], resolution=120)
        
        # Add glowing surface
        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            colorscale=[[0, colors[i]], [1, colors[i]]],
            showscale=False,
            opacity=0.7 - (i * 0.05),  # Gradually more transparent
            name=f'Shell {i+1}: R={Rn:.2f}',
            lighting=dict(
                ambient=0.6,
                diffuse=0.8,
                specular=1.0,
                roughness=0.1,
                fresnel=2.0
            ),
            lightposition=dict(x=100, y=100, z=100)
        ))
        
        print(f"  Shell {i+1}: R={Rn:.3f}, r={rn:.3f}, φ⁻{i} scaling")
    
    # Add central axis glow
    z_axis = np.linspace(-3, 3, 100)
    x_axis = np.zeros_like(z_axis)
    y_axis = np.zeros_like(z_axis)
    
    fig.add_trace(go.Scatter3d(
        x=x_axis, y=y_axis, z=z_axis,
        mode='lines',
        line=dict(color='white', width=10),
        showlegend=False
    ))
    
    # Add Flower of Life points (7 circles)
    for i in range(7):
        angle = i * 2 * np.pi / 7
        radius = R0 * 0.3
        x_fol = radius * np.cos(angle)
        y_fol = radius * np.sin(angle)
        
        fig.add_trace(go.Scatter3d(
            x=[x_fol], y=[y_fol], z=[0],
            mode='markers',
            marker=dict(
                size=15,
                color='gold',
                symbol='circle',
                line=dict(color='white', width=2)
            ),
            showlegend=False
        ))
    
    # Add quantum wave equation annotation
    fig.add_annotation(
        x=0.05, y=0.95,
        text="ψ(r,θ,φ) = R(r)Y<sub>l</sub><sup>m</sup>(θ,φ)",
        showarrow=False,
        xref="paper", yref="paper",
        font=dict(size=16, color="white", family="Arial"),
        bgcolor="rgba(0,0,0,0.5)",
        borderpad=5
    )
    
    # Add COSF value annotation
    fig.add_annotation(
        x=0.95, y=0.95,
        text="COSF = 42800/7.83  5466<br>φ  e (0.7% dev)",
        showarrow=False,
        xref="paper", yref="paper",
        font=dict(size=14, color="cyan", family="Courier"),
        bgcolor="rgba(0,0,0,0.7)",
        borderpad=5,
        align="right"
    )
    
    # Update layout for maximum impact
    fig.update_layout(
        title=dict(
            text='COSF Framework: Nested φ-Scaled Toroidal Resonance Shells',
            x=0.5,
            xanchor='center',
            font=dict(size=24, family='Arial Black', color='white')
        ),
        scene=dict(
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.2),
                center=dict(x=0, y=0, z=0)
            ),
            xaxis=dict(
                showbackground=False,
                showgrid=False,
                zeroline=False,
                visible=False
            ),
            yaxis=dict(
                showbackground=False,
                showgrid=False,
                zeroline=False,
                visible=False
            ),
            zaxis=dict(
                showbackground=False,
                showgrid=False,
                zeroline=False,
                visible=False
            ),
            bgcolor='rgb(10, 10, 30)'  # Deep space blue
        ),
        width=2400,
        height=1800,
        showlegend=False,
        paper_bgcolor='rgb(10, 10, 30)',
        plot_bgcolor='rgb(10, 10, 30)'
    )
    
    # Save in multiple formats
    print("\nSaving visualization...")
    
    # Interactive HTML
    fig.write_html('../../images/toroidal_geometry/ULTIMATE_COSF_VISUALIZATION.html')
    print(" Saved: ULTIMATE_COSF_VISUALIZATION.html")
    
    # Try to save static image
    try:
        fig.write_image('../../images/toroidal_geometry/ULTIMATE_COSF_VISUALIZATION.png',
                       width=2400, height=1800, scale=2)
        print(" Saved: ULTIMATE_COSF_VISUALIZATION.png (4K)")
    except:
        print(" Install kaleido for PNG: pip install kaleido")
    
    fig.show()
    
    print("\n ULTIMATE VISUALIZATION COMPLETE! ")

if __name__ == "__main__":
    create_ultimate_visualization()
