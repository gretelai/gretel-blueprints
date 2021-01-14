import random
from typing import List, Dict

import pandas as pd
import itertools


def get_mode_full_seeds(project_info: dict) -> List[dict]:
    """
    This function gets the smarts seeds needed to generate synthetic data
    when the user has chosen mode "full" (generate a full synthetic dataset).
    To get the number of synthetic lines needed per seed, it first compute
    the ratio needed for each field's values.  Then the global ratio needed
    per each combo seed is the product of each fields ratio.
    """

    even_percents = {}
    categ_val_lists = []
    seed_percent = 1
    balance_columns = []
    gen_lines = project_info["gen_lines"]
    for field in project_info["field_stats"]:
        if project_info["field_stats"][field]["use"]:
            values = set(pd.Series(project_info["records"][field].dropna()))
            category_cnt = len(values)
            even_percents[field] = 1/category_cnt
            categ_val_lists.append(list(values))
            seed_percent = seed_percent * even_percents[field]  
            balance_columns.append(field)

    seed_gen_cnt = seed_percent * gen_lines
    seed_fields = []
    for combo in itertools.product(*categ_val_lists):
        seed_dict = {}
        i = 0
        for field in balance_columns:
            seed_dict[field] = combo[i]
            i += 1
        seed = {}
        seed["seed"] = seed_dict
        seed["cnt"] = seed_gen_cnt
        seed_fields.append(seed)
       
    return seed_fields


def get_seed_amts(field: dict) -> Dict[str, int]:
    
    seed_needs = {}
    max = 0
    for class_value in field["class_cnts"]:
        if field["class_cnts"][class_value] > max:
            max = field["class_cnts"][class_value]
    for class_value in field["class_cnts"]:  
        seed_needs[class_value] = max - field["class_cnts"][class_value]
        
    return seed_needs


def get_mode_additive_seeds(project_info: dict) -> List[dict]:
    """
    This function gets the smarts seeds needed to generate synthetic data
    when the user has chosen mode "additive" (generate only enough synthetic data, such
    that when added to the original data creates a balanced set).
    """
    
    #First get the seed needs relative to that field
    seed_amts = {}
    for field in project_info["field_stats"]:
        if project_info["field_stats"][field]["use"]:
            seed_amts[field] = get_seed_amts(project_info["field_stats"][field])
            
    #Now determine the field with the highest need count, and bring the other
    #seeds up to that count, keeping each field balanced.
    
    max_need = 0
    field_needs = {}
    for field in seed_amts:
        need = 0
        for class_value in seed_amts[field]:
            need += seed_amts[field][class_value]
        field_needs[field] = need
        if need > max_need:
            max_need = need
                   
    for field in seed_amts:
        if field_needs[field] < max_need:
            diff = max_need - field_needs[field]
            more = True
            used = 0
            #Idea is to keep looping through the field's class values, incrementing the corresponding seed
            #amount by one each time, until the "diff" has been used up.  Once the "diff" is used up, we
            #must exit the loop immediately so that all field lists are of the same length.
            while more:
                for class_value in seed_amts[field]:
                    if used == diff:
                        more = False
                        continue
                    seed_amts[field][class_value] += 1
                    used += 1
    
    #Now create the needed combo seeds and their counts
    #For each field, we'll create a list of all values needed and sort it randomly.
    #The length of these lists will be the same, as we brought the sum of each field's
    #seed needs up to the max need.
    #Then we'll create combo seeds by taking one value from each field list
    
    field_lists = {}
    for field in seed_amts:
        curr_list = []
        for class_value in seed_amts[field]:
            for i in range(seed_amts[field][class_value]):
                curr_list.append(class_value)              
        random.shuffle(curr_list) 
        field_lists[field] = curr_list

    all_seeds = {}
    for i in range(max_need):
        seed_dict = {}
        seed = ""
        for field in field_lists:
            seed_dict[field] = field_lists[field].pop()
            seed = seed + "::" + seed_dict[field]
        
        if seed in all_seeds:
            all_seeds[seed]["cnt"] += 1
        else: 
            all_seeds[seed] = {}
            all_seeds[seed]["info"] = seed_dict
            all_seeds[seed]["cnt"] = 1
            
    #reformat to be like other mode seeds
    
    seed_fields = []
    for next in all_seeds:
        seed = {}
        seed["seed"] = all_seeds[next]["info"]
        seed["cnt"] = all_seeds[next]["cnt"]
        seed_fields.append(seed)
       
    return seed_fields


def gen_smart_seeds(project_info: dict) -> dict:

    if project_info["mode"] == "full":
        seeds = get_mode_full_seeds(project_info)
    else:
        seeds = get_mode_additive_seeds(project_info)
        
    project_info["seeds"] = seeds
    
    return project_info 


def compute_synth_needs(project_info: dict) -> dict:
    
    project_info = gen_smart_seeds(project_info)
    
    if project_info["mode"] == "additive":
        synth_needs = 0
        for seed in project_info["seeds"]:
            synth_needs += seed["cnt"]
            
        print("Total synthetic records required to fix bias is: " + str(synth_needs))
              
    return project_info


def gen_synth_nobias(model, project_info: dict) -> pd.DataFrame:
    """
    This is the main routine called in the synth auto-balance notebook for
    generating balanced synthetic data.  It returns the final synthetic
    dataframe.
    """

    seeds = project_info["seeds"]
    seed_cnt = len(seeds)
    bias_cnt = 0
    for field in project_info["field_stats"]:
        if project_info["field_stats"][field]["use"]:
            bias_cnt += 1
    
    print("Balancing synthetic generation for " + str(seed_cnt) + " value combinations from " + \
          str(bias_cnt) + " bias fields.\n")   
                         
    synth_df = pd.DataFrame(columns=project_info["records"].columns)
    
    max_invalid = 0
    if project_info["mode"] == "full":
        max_invalid = 1000 * project_info["gen_lines"]
    else:
        max_invalid = 1000 * project_info["num_records"]
 
    cnt = 1
    for seed in seeds:
        print("Balancing combination " + str(cnt) + " of " + str(seed_cnt) + ":")
        cnt += 1
        model.generate(num_lines=int(seed["cnt"]), max_invalid=max_invalid, seed_fields=seed["seed"])
        tempdf = model.get_synthetic_df()
        synth_df = synth_df.append(tempdf, ignore_index=True)

    return synth_df
