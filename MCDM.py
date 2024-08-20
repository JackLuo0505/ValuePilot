import numpy as np

def promethee_sort_actions_synthesis(results, preferences):
    """
    核心概念：
    PROMETHEE通过偏好函数对各方案在不同准则上的差异进行量化，计算出优势流和劣势流，再得出净流，
    最终对方案进行排序。

    优点：
    - 比AHP更灵活，允许选择不同的偏好函数来反映决策者的具体偏好。
    - 比TOPSIS更具解释性，明确展示了方案的相对优势和劣势。
    - 与MAUT类似，能够处理复杂的权重设置和多样的准则，适用于多属性决策。

    缺点：
    - 偏好函数选择复杂度较高，不如AHP直观。
    - 计算复杂性比TOPSIS和AHP高，尤其在处理大量方案时。
    - 对参数和权重的敏感度较高，可能需要多次调试以确保结果稳健。
    """
    num_actions = len(results)
    predicted_values_ratings = []
    scenario_predicted_ratings = []
    ratings = []

    def correcting_preference(x):
        return 1 / (1 + np.exp(-(x - 0.5 ) * 10))
    
    preferences = correcting_preference(preferences)
    
    # 比較與打分與preferences的差距
    # 计算每个动作的综合评分
    for sample in results:
        scenario_predicted_rating = np.array(sample['scenario_predictied_rating'])
        predicted_values_rating = np.array(sample['predicted_values_rating'])
        scenario_predicted_rating = correcting_preference(scenario_predicted_rating)
        predicted_values_rating = correcting_preference(predicted_values_rating)
        scenario_predicted_rating = (1 - np.abs(np.abs(scenario_predicted_rating) - preferences)) * 0.3
        predicted_values_rating = (1 - np.abs(np.abs(predicted_values_rating) - preferences)) * 0.3
        scenario_predicted_rating = scenario_predicted_rating + np.array(sample['scenario_predictied_rating']) * 0.7
        predicted_values_rating = predicted_values_rating + np.array(sample['predicted_values_rating']) * 0.7
        rating = 1 / (1 + np.exp( - np.abs(scenario_predicted_rating))) * predicted_values_rating
        
        scenario_predicted_ratings.append(scenario_predicted_rating)
        predicted_values_ratings.append(predicted_values_rating)
        rating = rating / np.sum(rating)
        ratings.append(rating)
    ratings = np.array(ratings)

    # 规范化权重
    preferences = preferences / np.sum(preferences)

    # 定义偏好函数
    def sigmoid_preference_function(option_a_score, option_b_score, k=1):
        diff = option_a_score - option_b_score
        return 1 / (1 + np.exp(-k * diff))

    # 计算偏好矩阵
    preference_matrix = np.zeros((num_actions, num_actions, len(rating)))
    for i in range(num_actions):
        for j in range(num_actions):
            for k in range(len(rating)):
                preference_matrix[i, j, k] = sigmoid_preference_function(ratings[i, k], ratings[j, k])

    # 使用加权乘积方法计算weighted_preferences
    weighted_preferences = np.ones((num_actions, num_actions))
    for k in range(len(preferences)):
        weighted_preferences *= preference_matrix[:, :, k] ** preferences[k]

    # 计算优势流和劣势流
    positive_flow = np.mean(weighted_preferences, axis=1)
    negative_flow = np.mean(weighted_preferences, axis=0)

    # 计算净流
    net_flow = positive_flow - negative_flow

    # 创建动作列表并附加评分信息
    sorted_actions = []
    for index, sample in enumerate(results):
        action_info = {
            'score': net_flow[index],  # 使用净流作为评分
            'index': index,
            'scenario predictied rating': scenario_predicted_ratings[index],
            'predicted values rating': predicted_values_ratings[index],
            'rating': ratings[index],
        }
        sorted_actions.append(action_info)

    # 按净流（score）对动作进行排序
    sorted_actions = sorted(sorted_actions, key=lambda x: x['score'], reverse=True)

    return sorted_actions

def ahp_sort_actions_synthesis(results, preferences):
    """
    核心概念：
    AHP通过将决策问题分解为层次结构（如目标、准则、子准则和备选方案），
    然后进行成对比较来确定各个准则和备选方案的相对重要性。决策者通过比较矩阵对各个元素进行打分，
    最终通过加权求和来确定最佳方案。
    
    优点(本次没有这些，见备註)：
    - 直观且结构化，适合处理复杂的决策问题。
    - 可以结合定性和定量因素。
    - 提供了一种一致性检查机制，帮助确保判断的一致性。
    缺点(本次没有这些，见备註)：
    - 成对比较的数量随着准则和备选方案数量的增加而显著增加，可能导致计算复杂性和判断疲劳。
    - 判断的主观性较强，可能导致结果偏差。
    
    备註:
    由于preferences直接由用户给出，而不需要用户制作比较矩阵，ahp变为简单的加权求和。
    """
    ratings = []
    predicted_values_ratings = []
    scenario_predicted_ratings = []

    def correcting_preference(x):
        return 1 / (1 + np.exp(-(x - 0.5 ) * 10))
    
    preferences = correcting_preference(preferences)

    # 比較與打分與preferences的差距
    # 计算每个动作的综合评分
    for sample in results:
        scenario_predicted_rating = np.array(sample['scenario_predictied_rating'])
        predicted_values_rating = np.array(sample['predicted_values_rating'])
        scenario_predicted_rating = correcting_preference(scenario_predicted_rating)
        predicted_values_rating = correcting_preference(predicted_values_rating)
        scenario_predicted_rating = (1 - np.abs(np.abs(scenario_predicted_rating) - preferences)) * 0.3
        predicted_values_rating = (1 - np.abs(np.abs(predicted_values_rating) - preferences)) * 0.3
        scenario_predicted_rating = scenario_predicted_rating + np.array(sample['scenario_predictied_rating']) * 0.7
        predicted_values_rating = predicted_values_rating + np.array(sample['predicted_values_rating']) * 0.7
        rating = 1 / (1 + np.exp( - np.abs(scenario_predicted_rating))) * predicted_values_rating
        
        scenario_predicted_ratings.append(scenario_predicted_rating)
        predicted_values_ratings.append(predicted_values_rating)
        rating = rating / np.sum(rating)
        ratings.append(rating)
    ratings = np.array(ratings)

    # 确保偏好（权重）总和为1，如果不是则归一化
    preferences = preferences / np.sum(preferences)

    # 计算各个动作的总得分
    weighted_scores = np.dot(ratings, preferences)

    # 将总得分附加到动作信息中
    sorted_actions = []
    for index, sample in enumerate(results):
        action_info = {
            'score': weighted_scores[index],  # 使用加权总得分作为评分
            'index': index,
            'scenario predictied rating': scenario_predicted_ratings[index],
            'predicted values rating': predicted_values_ratings[index],
            'rating': ratings[index],
        }
        sorted_actions.append(action_info)

    # 按总得分对动作进行排序
    sorted_actions = sorted(sorted_actions, key=lambda x: x['score'], reverse=True)

    return sorted_actions

def topsis_sort_actions(results, preferences):
    """
    核心概念：
    TOPSIS基于这样的假设，即最佳备选方案应距离理想解（最佳可能的解决方案）最近，
    同时距离负理想解（最差可能的解决方案）最远。通过计算每个备选方案与理想解和负理想解的欧几里得距离，
    来对备选方案进行排序。

    优点：
    - 简单且易于理解，计算步骤相对直接。
    - 不需要复杂的成对比较，计算量较小。
    - 结果具有较强的解释性，适合实际应用。
    缺点：
    - 依赖于标准化过程，可能会受到尺度选择的影响。
    - 假设准则之间是线性独立的，这在实际中不总是成立。
    """
    ratings = []
    predicted_values_ratings = []
    scenario_predicted_ratings = []

    def correcting_preference(x):
        return 1 / (1 + np.exp(-(x - 0.5 ) * 10))
    
    preferences = correcting_preference(preferences)

    # 比較與打分與preferences的差距
    # 计算每个动作的综合评分
    for sample in results:
        scenario_predicted_rating = np.array(sample['scenario_predictied_rating'])
        predicted_values_rating = np.array(sample['predicted_values_rating'])
        scenario_predicted_rating = correcting_preference(scenario_predicted_rating)
        predicted_values_rating = correcting_preference(predicted_values_rating)
        scenario_predicted_rating = (1 - np.abs(np.abs(scenario_predicted_rating) - preferences)) * 0.3
        predicted_values_rating = (1 - np.abs(np.abs(predicted_values_rating) - preferences)) * 0.3
        scenario_predicted_rating = scenario_predicted_rating + np.array(sample['scenario_predictied_rating']) * 0.7
        predicted_values_rating = predicted_values_rating + np.array(sample['predicted_values_rating']) * 0.7
        rating = 1 / (1 + np.exp( - np.abs(scenario_predicted_rating))) * predicted_values_rating
        
        scenario_predicted_ratings.append(scenario_predicted_rating)
        predicted_values_ratings.append(predicted_values_rating)
        rating = rating / np.sum(rating)
        ratings.append(rating)
    ratings = np.array(ratings)

    # 加权规范化矩阵
    weighted_ratings = ratings * preferences

    # 计算理想解和负理想解
    ideal_solution = np.max(weighted_ratings, axis=0)  # 理想解
    negative_ideal_solution = np.min(weighted_ratings, axis=0)  # 负理想解

    # 计算每个动作与理想解和负理想解的距离
    distance_to_ideal = np.sqrt(((weighted_ratings - ideal_solution)**2).sum(axis=1))
    distance_to_negative_ideal = np.sqrt(((weighted_ratings - negative_ideal_solution)**2).sum(axis=1))

    # 计算相似度得分（越接近1越好）
    similarity_to_ideal_solution = distance_to_negative_ideal / (distance_to_ideal + distance_to_negative_ideal)

    # 将相似度得分附加到动作信息中
    sorted_actions = []
    for index, sample in enumerate(results):
        action_info = {
            'score': similarity_to_ideal_solution[index],  # 使用相似度得分作为评分
            'index': index,
            'scenario predictied rating': scenario_predicted_ratings[index],
            'predicted values rating': predicted_values_ratings[index],
            'rating': ratings[index],
        }
        sorted_actions.append(action_info)

    # 按相似度得分对动作进行排序
    sorted_actions = sorted(sorted_actions, key=lambda x: x['score'], reverse=True)

    return sorted_actions

def exponential_utility_function(x, min_value, max_value, alpha=1):
    """
    指数效用函数用于将属性值 x 映射到 [0, 1] 区间，反映决策者的风险态度。
    
    - alpha > 0: 风险厌恶
        - 当 alpha 为正值时，函数呈现凹形，表明决策者对收益的增长逐渐感到满足。
        - 换句话说，随着 x 的增加，效用的增幅变得越来越小，意味着决策者更倾向于接受稳定的、确定的收益，而非高风险的回报。
        - 例如，风险厌恶者更愿意接受确定的小收益，而不愿意冒险去追求更大的潜在收益。
        
    - alpha = 0: 风险中立（线性）
        - 当 alpha 等于 0 时，指数效用函数退化为线性函数，表示决策者对风险持中立态度。
        - 在这种情况下，效用值与属性值成正比，决策者仅关心预期收益，而不关心收益的波动性。
        - 决策者会选择期望收益最高的方案，而不考虑方案之间的风险差异。
        
    - alpha < 0: 风险偏好
        - 当 alpha 为负值时，函数呈现凸形，表明决策者愿意为了获得潜在的高回报而承担较大的风险。
        - 在这种情况下，随着 x 的增加，效用的增幅变得越来越大，意味着决策者更愿意冒更大的风险，以期获得更高的回报。
        - 风险偏好者可能会选择波动性较大的方案，即使这些方案的预期收益可能与较稳定的方案相同或稍低。
        
    该函数通过调整 alpha 参数，可以灵活地模拟不同风险态度的决策者的行为，使得多准则决策分析（MCDA）能够更好地反映决策者的偏好。
    """
    normalized_x = (x - min_value) / (max_value - min_value)
    if alpha == 0:
        return normalized_x  # 线性映射
    else:
        return 1 - np.exp(-alpha * normalized_x)

def maut_sort_actions(results, preferences):
    """
    核心概念：
    MAUT（多属性效用理论）通过将各个准则的值转换为效用值，并根据决策者的偏好权重进行加权求和，
    计算每个方案的总效用值，从而进行排序和选择最优方案。

    优点：
    - 比AHP更适合处理带有不确定性和风险的决策问题，能够通过效用函数反映决策者的风险态度。
    - MAUT可以处理复杂的效用函数，使得在多准则决策中能够精细反映决策者的偏好。
    - 与TOPSIS相比，不依赖于理想解和负理想解的概念，允许更灵活的准则处理。

    缺点：
    - MAUT没有像PROMETHEE中的优势流和劣势流或TOPSIS中的距离概念，这可能导致在结果解释和稳健性分析上有所不足。
    - 构建效用函数的复杂度较高，要求决策者对各准则的效用理解深入，不如AHP直观。
    - 对决策者的要求较高，尤其在需要明确定义效用函数的情况下，可能增加主观性和使用难度。

    备註:
    在我们的项目中，风险现为价值评分的不确定性。每个动作都有多个价值相关的评分（如 predicted_values_rating 
    和 scenario_predicted_rating），这些评分可能并非确定值，而是基于预测结果得出的。
    因此，决策者可能面对的是价值评分的不确定性。
    但是，我们的排序应该总是风险厌恶或风险中立(比起单次的十分准确，我们更希望是多次的一般准确)。
    """
    ratings = []
    predicted_values_ratings = []
    scenario_predicted_ratings = []

    def correcting_preference(x):
        return 1 / (1 + np.exp(-(x - 0.5 ) * 10))
    
    preferences = correcting_preference(preferences)

    # 比較與打分與preferences的差距
    # 计算每个动作的综合评分
    for sample in results:
        scenario_predicted_rating = np.array(sample['scenario_predictied_rating'])
        predicted_values_rating = np.array(sample['predicted_values_rating'])
        scenario_predicted_rating = correcting_preference(scenario_predicted_rating)
        predicted_values_rating = correcting_preference(predicted_values_rating)
        scenario_predicted_rating = (1 - np.abs(np.abs(scenario_predicted_rating) - preferences)) * 0.3
        predicted_values_rating = (1 - np.abs(np.abs(predicted_values_rating) - preferences)) * 0.3
        scenario_predicted_rating = scenario_predicted_rating + np.array(sample['scenario_predictied_rating']) * 0.7
        predicted_values_rating = predicted_values_rating + np.array(sample['predicted_values_rating']) * 0.7
        rating = 1 / (1 + np.exp( - np.abs(scenario_predicted_rating))) * predicted_values_rating
        
        scenario_predicted_ratings.append(scenario_predicted_rating)
        predicted_values_ratings.append(predicted_values_rating)
        rating = rating / np.sum(rating)
        ratings.append(rating)
    ratings = np.array(ratings)

    min_values = np.min(ratings, axis=0)
    max_values = np.max(ratings, axis=0)

    # 计算每个动作在每个准则上的效用值
    utility_values = []
    for rating in ratings:
        utility = exponential_utility_function(rating, min_values, max_values, 0.5)
        utility_values.append(utility)
    utility_values = np.array(utility_values)

    # 计算加权效用值
    weighted_utility_values = utility_values * preferences

    # 计算综合效用值
    total_utility = np.sum(weighted_utility_values, axis=1)

    # 将总效用值附加到动作信息中
    sorted_actions = []
    for index, sample in enumerate(results):
        action_info = {
            'score': total_utility[index],  # 使用综合效用值作为评分
            'index': index,
            'scenario predictied rating': scenario_predicted_ratings[index],
            'predicted values rating': predicted_values_ratings[index],
            'rating': ratings[index],
        }
        sorted_actions.append(action_info)

    # 按总效用值对动作进行排序
    sorted_actions = sorted(sorted_actions, key=lambda x: x['score'], reverse=True)

    return sorted_actions

if '__main__' == __name__:
    # demo result for test
    result = [
        {
            'scenario_and_description': 'Action 0 Description',
            'scenario_predictied_rating': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            'initial_predicted_values_rating': [0, 0, 0, 0, 0, 0],
            'predicted_values_rating': [0.9, 0.1, 0.5, 0.5, 0.5, 0.5],
            'actual_values_rating': [0, 0, 0, 0, 0, 0],
            'initial_predicted_values_labeling': [0, 0, 0, 0, 0, 0],
            'predicted_values_labeling': [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            'actual_values_labeling': [0, 0, 0, 0, 0, 0],
        },
        {
            'scenario_and_description': 'Action 1 Description',
            'scenario_predictied_rating': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            'initial_predicted_values_rating': [0, 0, 0, 0, 0, 0],
            'predicted_values_rating': [0.7, 0.7, 0.3, 0.3, 0.5, 0.5],
            'actual_values_rating': [0, 0, 0, 0, 0, 0],
            'initial_predicted_values_labeling': [0, 0, 0, 0, 0, 0],
            'predicted_values_labeling': [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            'actual_values_labeling': [0, 0, 0, 0, 0, 0],
        },
        {
            'scenario_and_description': 'Action 2 Description',
            'scenario_predictied_rating': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            'initial_predicted_values_rating': [0, 0, 0, 0, 0, 0],
            'predicted_values_rating': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            'actual_values_rating': [0, 0, 0, 0, 0, 0],
            'initial_predicted_values_labeling': [0, 0, 0, 0, 0, 0],
            'predicted_values_labeling': [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            'actual_values_labeling': [0, 0, 0, 0, 0, 0],
        },
        {
            'scenario_and_description': 'Action 3 Description',
            'scenario_predictied_rating': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            'initial_predicted_values_rating': [0, 0, 0, 0, 0, 0],
            'predicted_values_rating': [1, 1, 1, -1, -1, -1],
            'actual_values_rating': [0, 0, 0, 0, 0, 0],
            'initial_predicted_values_labeling': [0, 0, 0, 0, 0, 0],
            'predicted_values_labeling': [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            'actual_values_labeling': [0, 0, 0, 0, 0, 0],
        },
    ]

    # demo preferences for test
    preferences = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    # Call the function with the mock data
    sorted_actions = ahp_sort_actions_synthesis(result, preferences)

    # Display the sorted actions
    for action in sorted_actions:
        print(f"Action Index: {action['index']}, Score: {action['score']:.4f}")
        print(f"Scenario Predicted Rating: {action['scenario predictied rating']}")
        print(f"Predicted Values Rating: {action['predicted values rating']}")
        print(f"Final Rating: {action['rating']}\n")

    """
    使用各种方法的排序:
    promethee_sort_actions_synthesis
    Action Index: 2, Score: 0.0538
    Action Index: 0, Score: 0.0366
    Action Index: 1, Score: 0.0270
    Action Index: 3, Score: -0.1175

    ahp_sort_actions_synthesis
    Action Index: 2, Score: 0.8559
    Action Index: 0, Score: 0.8215
    Action Index: 1, Score: 0.8016
    Action Index: 3, Score: 0.5000

    topsis_sort_actions
    Action Index: 2, Score: 0.8593
    Action Index: 1, Score: 0.7751
    Action Index: 0, Score: 0.7602
    Action Index: 3, Score: 0.2782

    maut_sort_actions
    Action Index: 2, Score: 0.8555
    Action Index: 0, Score: 0.8542
    Action Index: 1, Score: 0.7381
    Action Index: 3, Score: 0.5902

    根据这个例子比较片面的结果是四个方法都差不多，唯一可能就是promethee的区分度较大，可能还需要进一步的实验验证
    """