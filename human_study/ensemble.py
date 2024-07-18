import json
import os
from collections import defaultdict

def ensemble_rankings(folder_path):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.json')]
    
    # 初始化，存储每个场景每个动作的得分
    scores_by_person_scenario = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    # 遍历每个文件
    for file_path in files:
        with open(file_path, 'r') as file:
            data = json.load(file)
            for scenario in data['all_sorted_scenarios']:
                person_id = scenario['person_id']
                for i, rank_data in enumerate(scenario['sorted_scenarios']):
                    ranked_actions = rank_data['ranked_actions']
                    # 为每个动作赋予逆序分数
                    for index, action in enumerate(ranked_actions):
                        scores_by_person_scenario[person_id][i][action] += len(ranked_actions) - index

    # 构造最终结果
    final_results = {'all_sorted_scenarios': []}
    for person_id, scenarios in scores_by_person_scenario.items():
        sorted_scenarios = []
        for i, actions_scores in scenarios.items():
            sorted_actions = sorted(actions_scores.items(), key=lambda x: x[1], reverse=True)
            sorted_actions = [action for action, _ in sorted_actions]  # 提取动作名称
            sorted_scenarios.append({'ranked_actions': sorted_actions})
        final_results['all_sorted_scenarios'].append({
            'person_id': person_id,
            'sorted_scenarios': sorted_scenarios
        })

    # 写入文件
    with open(f'{folder_path}results_from_ensemble-model.json', 'w') as outfile:
        json.dump(final_results, outfile, indent=4)

# 调用函数
ensemble_rankings('human_study/ValuePilot/')
