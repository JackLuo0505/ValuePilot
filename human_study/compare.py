import json
import statistics
import argparse
import os

def consolidate_order_records(encoder_model):
    input_directory_path = f'human_study/ValuePilot/{encoder_model}'
    output_directory_path = 'human_study/ValuePilot'
    output_filename = f'results_from_{encoder_model}.json'
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
    # 初始化变量，用于存储每个深度下的交集比例
    depth_similarity = []

    # 遍历不同的深度
    for depth in range(1, min(len(list1), len(list2)) + 1):
        # 提取前depth个元素组成的子列表
        sublist1 = list1[:depth]
        sublist2 = list2[:depth]

        # 将子列表转换为集合
        set1 = set(sublist1)
        set2 = set(sublist2)

        # 计算交集大小
        intersection = len(set1.intersection(set2))

        # 计算交集大小相对于当前深度的比例
        similarity = intersection / depth

        # 将交集比例存储到列表中
        depth_similarity.append(similarity)

    # 计算平均相似度
    avg_similarity = sum(depth_similarity) / len(depth_similarity)

    return avg_similarity

def compute_similarities(file1, file2):
    # 读取两个JSON文件
    with open(file1, 'r') as f:
        data1 = json.load(f)
    with open(file2, 'r') as f:
        data2 = json.load(f)
    
    # 获取排序列表，并将 lists1 中的字符串转换为整数
    all_lists1 = [            
        [[int(action) for action in scenario['ranked_actions']] for scenario in person['sorted_scenarios']]
        for person in data1['all_sorted_scenarios']
    ]
    all_lists2 = [group['choices'] for group in data2]

    # 确保处理过程中不会超出索引范围
    min_len = min(len(all_lists1), len(all_lists2))

    # 存储所有人的相似度
    all_similarities = []

    # 计算每对排序列表的相似度
    for person_idx in range(min_len):
        person_lists1 = all_lists1[person_idx]
        person_lists2 = all_lists2[person_idx]
        
        # 确保每个人的列表数量相同
        scenario_len = min(len(person_lists1), len(person_lists2))
        
        similarities = []
        for scenario_idx in range(scenario_len):
            list1 = person_lists1[scenario_idx]
            list2 = person_lists2[scenario_idx]
            similarity = set_based_measure(list1, list2)
            similarities.append(similarity)
        
        # 计算并输出每个person_idx的相似度平均值
        avg_similarity = sum(similarities) / len(similarities)
        all_similarities.append(avg_similarity)
        print(f"Person {person_idx+1} average similarity: {avg_similarity}")

    # 计算所有人的相似度平均值和标准差
    overall_avg_similarity = sum(all_similarities) / len(all_similarities)
    overall_std_dev_similarity = statistics.stdev(all_similarities)
    print(f"Overall average similarity: {overall_avg_similarity}")
    print(f"Overall standard deviation: {overall_std_dev_similarity}")

if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--encoder_model', type=str, default='t5-base', help='Encoder model to use', 
                        choices=['t5-small', 't5-base', 'flan-t5-base', 'bert-base-uncased', 'roberta-base'])
    encoder_model = parser.parse_args().encoder_model
    
    consolidate_order_records(encoder_model)
    # 调用函数计算相似度
    print("similarity_model:")
    compute_similarities(f'human_study/ValuePilot/results_from_{encoder_model}.json', 'human_study/result_en.json')
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
