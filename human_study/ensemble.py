import json
import os
from collections import defaultdict
from compare import compute_similarities

def ensemble_rankings(folder_path):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.json')]
    
    # initialize a dictionary to store scores by person and scenario
    scores_by_person_scenario = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    # iterate over the files
    for file_path in files:
        with open(file_path, 'r') as file:
            data = json.load(file)
            for scenario in data['all_sorted_scenarios']:
                person_id = scenario['person_id']
                for i, rank_data in enumerate(scenario['sorted_scenarios']):
                    ranked_actions = rank_data['ranked_actions']
                    # update the scores for each action in the ranked list
                    for index, action in enumerate(ranked_actions):
                        scores_by_person_scenario[person_id][i][action] += len(ranked_actions) - index
                        if index == 0:
                            scores_by_person_scenario[person_id][i][action] += 2
                        elif index == 1:
                            scores_by_person_scenario[person_id][i][action] += 1
                        elif index == len(ranked_actions) - 1:
                            scores_by_person_scenario[person_id][i][action] -= 2
                        elif index == len(ranked_actions) - 2:
                            scores_by_person_scenario[person_id][i][action] -= 1
                        
    # calculate the final scores
    final_results = {'all_sorted_scenarios': []}
    for person_id, scenarios in scores_by_person_scenario.items():
        sorted_scenarios = []
        for i, actions_scores in scenarios.items():
            sorted_actions = sorted(actions_scores.items(), key=lambda x: x[1], reverse=True)
            sorted_actions = [action for action, _ in sorted_actions]  # extract the actions
            sorted_scenarios.append({'ranked_actions': sorted_actions})
        final_results['all_sorted_scenarios'].append({
            'person_id': person_id,
            'sorted_scenarios': sorted_scenarios
        })

    # save the final results to a new JSON file
    with open(f'{folder_path}results_from_ensemble-model.json', 'w') as outfile:
        json.dump(final_results, outfile, indent=4)

# Call the function with the path to the directory containing the JSON files
ensemble_rankings('human_study/ValuePilot/')

print("ensemble:")
compute_similarities('human_study/ValuePilot/results_from_ensemble-model.json', 'human_study/result_en.json')
print("\n")