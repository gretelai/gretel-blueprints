import os
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from gretel_auto_xf.facts import ProjectFacts
from gretel_auto_xf.pipeline import Config
from gretel_client.projects import Project

F_RATIO_THRESHOLD = .7


def get_field_type(types: List[dict]) -> Optional[str]:
    """
    If the field contains any numeric values, return numeric.  
    We want to return type string only when the field contains strings
    but no numeric.
    """

    found_categorical = False
    for next_type in types:
        if next_type["type"] == "numeric":
            return "numeric"
        if next_type["type"] == "string":
            found_categorical = True
            
    if found_categorical:
        return "string"
    else:
        return None

    
def get_entities(entities: dict) -> List[str]:
    """
    We only want to list an entity with a field if it is pervasively 
    tagged in the column.
    """
    
    entity_list = []
    for next_entity in entities:
        if next_entity["f_ratio"] > F_RATIO_THRESHOLD:
            entity_list.append(next_entity["label"])
            
    return entity_list


def get_distrib(class_cnts: Dict[str, int], count: int) -> Dict[str, float]:
  
    distribution = {}
    for k in class_cnts.keys():
        distribution[k] = class_cnts[k] / count
        
    sorted_distrib = {k: v for k, v in sorted(distribution.items(), key=lambda item: item[1], reverse=True)}
        
    return sorted_distrib


def get_field_cnts(field: pd.Series) -> Dict[str, int]:
    
    distribution = {}
    field_clean = field.dropna()
    for v in field_clean:
        distribution[str(v)] = distribution.get(str(v), 0) + 1
    
    return distribution


def get_project_facts(project: Project, num_records: int) -> "ProjectFacts": 
    """
    This function borrows a method from the gretel_auto_xf module to retrieve
    information about a given project
    """
    
    config = Config() 
    config.max_records = num_records
        
    return ProjectFacts.from_project(project, config)


def get_project_info(project: Project, mode = "full", num_records = 5000, gen_lines = None) -> dict:
    """
    This gathers the necessary information from a Project to support synthetic auto-balance
    
    Arguments:
        project: Reference to a project's API client
        mode: Can be either "full" or "additive".  Mode "full" means generate a full synthetic balanced dataset.
              Mode "additive" means only generate enough synthetic samples, such that when added to the
              original set, the categorical classes are balanced.
        num_records: How many records to retrieve from the project.
        gen_lines: In mode "full", this is the number of synthetic records you'd like generated.
        
    Returns:
        A data structure that is used throughout the synthetic auto-balance notebook   
    """
    
    project_info = {}
    facts = get_project_facts(project, num_records)
    project_info["mode"] = mode
    project_info["gen_lines"] = gen_lines
    project_info["records"] = facts.as_df
    project_info["num_records"] = len(project_info["records"].index)
    project_info["field_stats"] = {}
    
    for field in facts.stats:
        if get_field_type(facts.stats[field]["types"]) != "string":
            continue
        entities = get_entities(facts.stats[field]["entities"])
        if "date" in entities:
            continue
        project_info["field_stats"][field] = {}
        project_info["field_stats"][field]["count"] = facts.stats[field]["count"]
        project_info["field_stats"][field]["cardinality"] = facts.stats[field]["approx_cardinality"]
        project_info["field_stats"][field]["pct_missing"] = facts.stats[field]["pct_missing"]
        project_info["field_stats"][field]["use"] = False
        project_info["field_stats"][field]["entities"]  = entities
        project_info["field_stats"][field]["class_cnts"] = get_field_cnts(project_info["records"][field])
        project_info["field_stats"][field]["distrib"] = get_distrib(project_info["field_stats"][field]["class_cnts"],
                                                                    project_info["field_stats"][field]["count"])
    
    return project_info


def bias_fields(project_info: dict) -> List[str]:
    """
    This functions returns the list of fields that were chosen
    by the user in the notebook to remove bias from
    """
    
    use_fields = []
    for field in project_info["field_stats"]:
        if project_info["field_stats"][field]["use"]:
            use_fields.append(field)

    return use_fields
