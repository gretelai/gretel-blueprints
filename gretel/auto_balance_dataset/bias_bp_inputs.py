from tabulate import tabulate
from ipywidgets import HBox, VBox, widgets, Layout, HTML
from functools import partial


def choose_bias_fields(project_info: dict) -> dict:
    """
    This is the function called from the synthetic auto-balance notebook that
    enables the user to pick which fields they want to remove bias from.
    It displays a table consisting of the categorical fields in the project,
    and for each field shows its cardinality, %missing and
    prevalent entities associated with the column.
    Approach/code hugely borrowed from the gretel_auto_xf module.
    """
    
    table = {}
    field_names = []
    field_cardinality = []
    field_pct_missing = []
    field_entities = []
    for field in project_info["field_stats"]:
        field_names.append(field)
        field_cardinality.append(project_info["field_stats"][field]["cardinality"])
        field_pct_missing.append(project_info["field_stats"][field]["pct_missing"])
        field_entities.append(project_info["field_stats"][field]["entities"])
    table["Field"] = field_names
    table["Unique Value Cnt"] = field_cardinality
    table["% Missing"] = field_pct_missing
    table["Entities"] = field_entities
    
    report_str = tabulate(table, headers="keys", tablefmt="simple")

    line_height = "1.5rem"
    padding_margin = {"margin": "0 0 0 0", "padding": "0 0 0 0"}

    layout = Layout(width="30px", height=line_height, **padding_margin)  # type: ignore

    def on_check(field, evt: dict):
        if evt.get("new"):
            project_info["field_stats"][field]["use"] = True
        else:
            project_info["field_stats"][field]["use"] = False

    def build_checkbox(field):
        check_box = widgets.Checkbox(value=False, indent=False, layout=layout)  # type: ignore
        check_box.observe(partial(on_check, field), names=["value"])
        return check_box

    buttons = [build_checkbox(field) for field in project_info["field_stats"]]

    display(
        HBox(
            [
                VBox(
                    [
                        widgets.Label(layout=layout),  # type: ignore
                        widgets.Label(layout=layout),  # type: ignore
                        *buttons,
                    ],
                    layout=Layout(**padding_margin),  # type: ignore
                ),
                HTML(f'<pre style="line-height:{line_height}">{report_str}</pre>'),
            ],
            layout=Layout(**padding_margin),  # type: ignore
        )
    )
    
    return(project_info)