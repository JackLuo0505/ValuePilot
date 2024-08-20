import json
import statistics
import argparse
import os

def consolidate_order_records(input_pth, MCDM_method):
    input_directory_path = f'{input_pth}{MCDM_method}'
    output_directory_path = input_pth
    output_filename = f'results_from_{MCDM_method}.json'
    all_data = {"all_sorted_scenarios": []}

    # Loop through files in the directory
    for filename in os.listdir(input_directory_path):
        if filename.startswith("order_record_") and filename.endswith(".json"):
            file_path = os.path.join(input_directory_path, filename)

            # Open and read the JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)
                person_id = int(filename.split('_')[-1].split('.')[0])  # Extract person_id from filename

                # Process each record in the JSON file
                person_record = {"person_id": person_id, "sorted_scenarios": []}
                for item in data:
                    sorted_scenario = {
                        "ranked_actions": [str(num) for num in item["model output order"]]
                    }
                    person_record["sorted_scenarios"].append(sorted_scenario)
                
                all_data["all_sorted_scenarios"].append(person_record)

    # Sort data by person_id before writing to file
    all_data["all_sorted_scenarios"].sort(key=lambda x: x["person_id"])

    # Write the consolidated data to a new JSON file in the specified output directory
    with open(os.path.join(output_directory_path, output_filename), 'w') as output_file:
        json.dump(all_data, output_file, indent=4)

def set_based_measure(list1, list2):
    depth_similarity = []

    # iterate over the depth of the lists
    for depth in range(1, min(len(list1), len(list2)) + 1):
        # get the sublist of each list up to the current depth
        sublist1 = list1[:depth]
        sublist2 = list2[:depth]

        # transform the sublists into sets
        set1 = set(sublist1)
        set2 = set(sublist2)

        # compute the size of the intersection
        intersection = len(set1.intersection(set2))

        # compute the similarity as the size of the intersection divided by the depth
        similarity = intersection / depth

        # store the similarity
        depth_similarity.append(similarity)

    # compute the average similarity
    avg_similarity = sum(depth_similarity) / len(depth_similarity)

    return avg_similarity

def compute_similarities(file1, file2):
    # load the data from the files
    with open(file1, 'r') as f:
        data1 = json.load(f)
    with open(file2, 'r') as f:
        data2 = json.load(f)
    
    # extract the lists of actions for each person
    all_lists1 = [            
        [[int(action) for action in scenario['ranked_actions']] for scenario in person['sorted_scenarios']]
        for person in data1['all_sorted_scenarios']
    ]
    all_lists2 = [group['choices'] for group in data2]

    # find the minimum length of the two lists
    min_len = min(len(all_lists1), len(all_lists2))

    # initialize a list to store the similarities for each person
    all_similarities = []

    # iterate over the people
    for person_idx in range(min_len):
        person_lists1 = all_lists1[person_idx]
        person_lists2 = all_lists2[person_idx]
        
        # find the minimum length of the two lists
        scenario_len = min(len(person_lists1), len(person_lists2))
        
        similarities = []
        for scenario_idx in range(scenario_len):
            list1 = person_lists1[scenario_idx]
            list2 = person_lists2[scenario_idx]
            similarity = set_based_measure(list1, list2)
            similarities.append(similarity)
        
        # compute the average similarity for the person
        avg_similarity = sum(similarities) / len(similarities)
        all_similarities.append(avg_similarity)
        print(f"Person {person_idx+1} average similarity: {avg_similarity}")

    # compute the overall average similarity and standard deviation
    overall_avg_similarity = sum(all_similarities) / len(all_similarities)
    overall_std_dev_similarity = statistics.stdev(all_similarities)
    print(f"Overall average similarity: {overall_avg_similarity}")
    print(f"Overall standard deviation: {overall_std_dev_similarity}")

if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--MCDM_method', type=str, default='ahp', help='MCDM method used',
                        choices=['ahp', 'topsis', 'maut'])
    MCDM_method = parser.parse_args().MCDM_method
    input_pth = 'human_study/Ablation_Experiment_MCDM/'
    
    consolidate_order_records(input_pth, MCDM_method)
    # compute similarities
    print("similarity_model:")
    compute_similarities(f'{input_pth}results_from_{MCDM_method}.json', 'human_study/result_en.json')
    print("\n")
    print("GPT-4o:")
    compute_similarities('human_study/benchmarks/gpt4.json', 'human_study/result_en.json')
    print("\n")
    print("gemini:")
    compute_similarities('human_study/benchmarks/gemini.json', 'human_study/result_en.json')
    print("\n")
    print("claude:")
    compute_similarities('human_study/benchmarks/claude.json', 'human_study/result_en.json')
    print("\n")
    print("llama:")
    compute_similarities('human_study/benchmarks/llama.json', 'human_study/result_en.json')