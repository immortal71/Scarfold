import plotly.graph_objects as go
import numpy as np

def plot_structure(coords, title='structure', save_html=None, show=True, colors=None):
    # coords: (L,3)
    L = coords.shape[0]
    x,y,z = coords[:,0], coords[:,1], coords[:,2]
    fig = go.Figure()
    # backbone as lines
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines+markers',
                               marker=dict(size=4, color='blue'),
                               line=dict(width=3, color='darkblue')))
    if colors is not None:
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers',
                                   marker=dict(size=6, color=colors, colorscale='Viridis', showscale=True)))
    fig.update_layout(title=title, width=800, height=700,
                      scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'))
    if save_html:
        fig.write_html(save_html)
    if show:
        fig.show()
    return fig

def plot_pred_and_native(pred_coords, true_coords, save_html=None):
    # overlay both
    L = pred_coords.shape[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=true_coords[:,0], y=true_coords[:,1], z=true_coords[:,2],
                               mode='lines+markers', name='native', marker=dict(size=3,color='green'),
                               line=dict(width=3,color='green')))
    fig.add_trace(go.Scatter3d(x=pred_coords[:,0], y=pred_coords[:,1], z=pred_coords[:,2],
                               mode='lines+markers', name='predicted', marker=dict(size=3,color='red'),
                               line=dict(width=3,color='red')))
    fig.update_layout(width=900, height=700)
    if save_html:
        fig.write_html(save_html)
    fig.show()

def plot_contact_map(pred_dists, true_dists=None, save_html=None, title='contact-map'):
    # show predicted distance heatmap, optionally with true-overlay diff
    l = pred_dists.shape[0]
    # convert to contact probabilities (lower is stronger contact)
    z = np.exp(-pred_dists / 8.0)
    fig = go.Figure(data=go.Heatmap(z=z, x=list(range(l)), y=list(range(l)), colorscale='Viridis'))
    fig.update_layout(title=title, width=650, height=620,
                      xaxis_title='residue i', yaxis_title='residue j')
    if save_html:
        fig.write_html(save_html)
    fig.show()
    if true_dists is not None:
        delta = np.abs(pred_dists - true_dists)
        fig2 = go.Figure(data=go.Heatmap(z=delta, x=list(range(l)), y=list(range(l)), colorscale='RdBu', zmid=0.0))
        fig2.update_layout(title=title + ' difference = |pred - truth|', width=650, height=620,
                           xaxis_title='residue i', yaxis_title='residue j')
        if save_html:
            fig2.write_html(save_html.replace('.html', '_delta.html'))
        fig2.show()


def plot_plddt(plddt_scores, save_html=None, title='pLDDT profile'):
    L = len(plddt_scores)
    fig = go.Figure(data=go.Scatter(x=list(range(L)), y=plddt_scores, mode='lines+markers', line=dict(color='dodgerblue')))
    fig.update_layout(title=title, xaxis_title='residue', yaxis_title='pLDDT-like score (0-100)', width=800, height=420)
    if save_html:
        fig.write_html(save_html)
    fig.show()


def plot_tm_score(tm, save_html=None, title='TM-score proxy'):
    fig = go.Figure(data=go.Indicator(mode='gauge+number', value=tm,
                                       gauge={'axis': {'range': [0, 1]}, 'bar': {'color': 'darkblue'}}))
    fig.update_layout(title=title, width=500, height=400)
    if save_html:
        fig.write_html(save_html)
    fig.show()