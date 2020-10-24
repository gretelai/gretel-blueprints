import math

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd

_GRETEL_PALETTE = ['#A051FA', '#18E7AA']
_GRAPH_OPACITY = 0.75
_GRAPH_BARGAP = 0.2  
_GRAPH_BARGROUPGAP = .1 
_GRAPH_MAX_BARS = 1000


def get_graph_dimen(fields: dict, uniq_cnt_threshold: int):
    """
    Helper function to first figure out how many graphs we'll be
    displaying, and then based on that determine the appropriate
    row and column count for display
    """

    graph_cnt = 0
    for field in fields:
        if fields[field]["cardinality"] <= uniq_cnt_threshold:
            graph_cnt += 1
    
    col_cnt = 1    
    if uniq_cnt_threshold <= 50:
        col_cnt = min(3, graph_cnt)
    elif uniq_cnt_threshold <= 100:
        col_cnt = min(2, graph_cnt)
    else:
        col_cnt = 1
    row_cnt = math.ceil(graph_cnt/col_cnt)    
            
    return row_cnt, col_cnt

 
def get_distrib_show(distrib: dict) -> dict:
    """
    Plotly slighly freaks with more than 1000 bars, so in the remote
    chance they chose to see graphs with more than 1000 unique values
    limit the graph bars to the highest 1000 values
    """
 
    if len(distrib) <= _GRAPH_MAX_BARS:
        return(distrib)

    cnt = 0
    new_distrib = {}
    for field in distrib:
        new_distrib[field] = distrib[field]
        cnt += 1
        if cnt == _GRAPH_MAX_BARS:
            return(new_distrib) 
    
    
def show_field_graphs(fields: dict, uniq_cnt_threshold=10):
    """
    This function takes the categorical fields in a project that have
    a unique value count less than or equal to the parameter
    "uniq_cnt_threshold" and displays their current distributions 
    using plotly bar charts. The number of columns used to display
    the graphs will depend on this value as well.
    """

    row_cnt, col_cnt = get_graph_dimen(fields, uniq_cnt_threshold)
    titles = []
    for field in fields:
        if fields[field]["cardinality"] <= uniq_cnt_threshold:
            titles.append(field)
    
    shared_yaxes = True
    if (col_cnt == 1):
        shared_yaxes = False
        
    fig = make_subplots(rows=row_cnt, cols=col_cnt, shared_yaxes=shared_yaxes, subplot_titles=titles)
                       
    row = 1
    col = 1
    for field in fields:
        if fields[field]["cardinality"] <= uniq_cnt_threshold:
            distrib = get_distrib_show(fields[field]["distrib"]) 
            fig.add_trace(
                go.Bar(
                    x=list(distrib.keys()), 
                    y=list(distrib.values()),
                    name=field
                ), 
                row, 
                col
            )
            if col == col_cnt:
                col = 1
                row += 1
            else:
                col += 1
            
    height = (700 / col_cnt) * row_cnt
            
    fig.update_layout(
        height=height, 
        width=900,
        showlegend=False,
        title='<span style="font-size: 16px;">Existing Categorical Field Distributions</span>',
        font=dict(
            size=8,
            color="RebeccaPurple"
        )   
    )
    fig.show()

    
def get_new_distrib(field: pd.Series) -> dict:
    """
    Even though we know what the new distribution will be, here
    we compute it fresh from the new data as a sanity check
    """
    
    distribution = {}
    for v in field:
        distribution[str(v)] = distribution.get(str(v), 0) + 1
    series_len = float(len(field))
    for k in distribution.keys():
        distribution[k] = distribution[k] / series_len
        
    return distribution   


def show_bar_chart(orig: dict, new: dict, field: str, mode: str):
    """
    This function takes two distributions (orig and new), along
    with the name of the field and mode and plots the 
    distributions on the same graph
    """

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=list(orig.keys()),
            y=list(orig.values()),
            name='Training',
            marker_color=_GRETEL_PALETTE[0],
            opacity=_GRAPH_OPACITY
        )
    )
    
    name = "Synthetic"
    if mode == "additive":
        name = "Training + Synthetic"
        
    fig.add_trace(
        go.Bar(
            x=list(new.keys()),
            y=list(new.values()),
            name=name,
            marker_color=_GRETEL_PALETTE[1],
            opacity=_GRAPH_OPACITY
        )
    )
    fig.update_layout(
        title='<span style="font-size: 16px;">Field: ' + field + '</span>',
        yaxis_title_text='Percentage',
        bargap=_GRAPH_BARGAP,
        bargroupgap=_GRAPH_BARGROUPGAP,
        barmode='group'
    )
    fig.show()

    
def show_new_graphs(project_info: dict, synth_df: pd.DataFrame):
    """
    This function is called at the conclusion of the synth auto-balance notebook to take
    a look at how the new distributions compare to the original
    """
    
    new_df = pd.DataFrame()
    if project_info["mode"] == "additive":
        new_df = pd.concat([project_info["records"], synth_df], ignore_index=True)
    else:
        new_df = synth_df
        
    for field in project_info["field_stats"]:
        if project_info["field_stats"][field]["use"]:
            new = pd.Series(new_df[field]).dropna()           
            new_distrib = get_new_distrib(new)
            show_bar_chart(project_info["field_stats"][field]["distrib"], new_distrib, field, project_info["mode"])

  
    
    