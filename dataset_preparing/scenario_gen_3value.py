import itertools
import json
import os
import sys
from openai import AzureOpenAI
import time
import re
import threading
from tqdm import tqdm


REGION = "xxxxx"
MODEL = "xxxxx"
API_KEY = "xxxxx"
API_BASE = "xxxxx"
ENDPOINT = f"{API_BASE}/{REGION}" 

client = AzureOpenAI(
    api_key = API_KEY,
    api_version = "2024-02-01",
    azure_endpoint = ENDPOINT,
)

def generate_all(values, desciptions, model = MODEL, temperature=1, max_tokens=2000):
    response = client.chat.completions.create(
        model=model,
        messages=[{
            'role': 'system', 
            'content': f"""This is a scenario generation task. 
                        The goal is to train an intelligent human-like agent's value-driven decision model, aspiring to humanize the agent in decision-making processes. 
                        We focus on values including 'curiosity', 'energy', 'safety', 'happiness', 'intimacy' and 'fairness'.
                        I want you to act as a sociologist and screenwriter with a sufficient understanding of the behavioral motivation and the value drive behind the interaction between people in society.
                        You will come up with interesting characters, actually human-like agents, the setting of the encounter, dialogues and interactions between the characters etc.
                        YOU MUST REMEMBER THAT THE CHARACTERS ARE REAL HUMAN BEINGS IN THE SCENARIO (I DON'T WANT TO SEE YOU CHOOSE AN ARTEFACT OR A ROBOT AS AN AGENT).
                        The interaction can happen everywhere reasonable, so open your imagination.
                        Please help me in the dataset preparing phase."""},

            {'role': 'user',
            'content': f"""
1. Scenario Generation
Create a scenario within a household where 2-4 agents interact with each other or one singal agents interact with the environment around him. Ensure that the agents' actions will embody {values[0]} and {values[1]} and {values[2]}, while DO NOT embody other basic values mentioned before.
{values[0]} means {desciptions[0]}.
{values[1]} means {desciptions[1]}.
{values[2]} means {desciptions[2]}.
You must remember that the characters are real human-being in the scenario(I don't want to see you choose an artefact or a robot as an agent), and the background of the interaction story between the protagonist and other characters is set in a home environment.
Designate the protagonist as 'Agent A', and label other agents as 'Agent B', 'Agent C', etc.
YOU MUST REMEMBER THAT THE CHARACTERS ARE REAL HUMAN BEINGS IN THE SCENARIO (I DON'T WANT TO SEE YOU CHOOSE AN ARTEFACT OR A ROBOT AS AN AGENT).

The scenario you generate is a point in time where some further behavior of the agents may occur, so your scene description should pave the way for the next possible actions of agents.

Additionally, it requires rich scene details, and requires that after reading the scenario in the first part, the reader can have a general prediction of the action decision of A.
It is not required to read the scenario to have a particularly clear corresponding action, but at least there is a clue.

BUT YOU MUSTN'T SAY HERE THAT THE PROTAGONIST ARE EQUIPPED WITH THESE TWO VALUES WHEN YOU DESCRIBE THE IDENTITIES OF THE AGENTS, BECAUSE WE WANT OUR MODEL TO LEARN THE VALUE DRIVE BEHIND THE PROTAGONIST'S ACTIONS THROUGH THE GENERATED SCENARIO.
ONLY INTRODUCE THE AGENTS' IDENTITIES AND DON'T INTRODUCE THE AGENTS' CHARACTER, THESE TWO WORDS DESCRIBING VALUES AND ALL OF THEIR SYNONYMS AND SAME-ROOTED WORDS CAN'T EVEN APPEAR HERE APPARENTLY.
AND DO NOT LET ME SEE ANY OF THOSE TWO WORDS AND ALL OF THEIR SYNONYMS AND SAME-ROOTED WORDS WHEN ENRICHING YOUR DETAILS.

AFTER YOU GENERATE YOUR TEXT, YOU MUST CHECK OUT IF ANY SENTENCES INCLUDE {values[0]} OR {values[1]} OR {values[2]} OR ALL OF THEIR SYNONYMS OR SAME-ROOTED WORDS. IF SO, DELETE THOSE SENTENCES.

2. Action Listing
List at least ten diverse actions that the protagonist 'Agent A' can take in the scenario.
REMEMBER THAT THE ACTIONS MUST BE MADE BY AGENT A.
The agent A's actions should embody out {values[0]} and {values[1]} and {values[2]}.
The actions here must be actions happening after the scenerio discribed above and must be reasonable and predictable after we read the scenerio above.
The actions here mustn't include other persons unless the other persons are the other agents.
The actions here must be highly relevant to the scenario you generate above.
It is not required to read the scenario to make readers have a particularly clear corresponding action, but at least there is a clue.
Make sure these actions reflect varying emphasis on the chosen values.
DO NOT CHOOSE TYPICALLY NON-SUBSTANTIVE ACTIONS SUCH AS 'SUPPOSE' AND 'PROPOSE'.
Label these actions as 'actions.'
Actions should reflect positive or negative aspects of each value. 
Ensure that in these actions, each value has the potential for positive and negative dimensions.
For instance, in a dining scenario, sharing food with others might be a negative action in terms of 'energy' because it depletes one's own energy. 
However, in the context of 'intimacy', it could be a positive action as it contributes to fostering a sense of closeness.
Please create actions in a similar manner, ensuring they can manifest both positive and negative aspects under different perspectives.
AFTER YOU GENERATE ACTIONS, YOU MUST CHECK OUT IF ANY ACTION DESCRIPTIONS INCLUDE {values[0]} OR {values[1]} OR {values[2]} OR ALL OF THEIR SYNONYMS OR SAME-ROOTED WORDS. IF SO, DELETE THOSE ACTIONS AND CREATE NEW ONES.

3. Value Evaluation
Independently identify and label the values exhibited in the scenario and actions, represented as 'values(evaluation)'.

4. Training Suitability
The scenario should comply with the following rules:
1. The scenario is suitable for training agents' value modules.
2. The scenario DO NOT contain other values besides {values[0]} and {values[1]} and {values[2]}.
3. The scenario is conducive to interactive display and engagement within an experimental environment.
IF THE SCENARIO IS SUITABLE FOR TRAINING, PLEASE OUTPUT 'YES', OTHERWISE OUTPUT 'NO' AND GO BACK TO STEP1.

5. Action Scoring
If the scenario is suitable for training, assign scores to each action on a scale of -1.0 to 1.0 on {values[0]}, {values[1]}, {values[2]}.
Both {values[0]} and {values[1]} and {values[2]} need to be scored, and they should not be zero, as the generated scenarios are closely related to {values[0]} and {values[1]} and {values[2]}.
Positive Value: Indicates that an action has a favorable impact on the specified value. 
For example, in 'happiness,' a positive value might signify that the action positively influences the sense of joy.
Negative Value: Indicates that an action has an unfavorable impact on the specified value. 
For instance, in 'safety,' a negative value could indicate that the action negatively affects the sense of security.
Such scoring reflects the effects of each action under different values, 
aiding in understanding how actions influence the positive or negative attributes of specific values.
Evaluate their alignment with each of the 'values(generation)' separately.
Ensure the scoring methodology is consistent and justified.

Here, I provide you with an output example, but please refrain from outputting this again for me. Just use it as a reference for the output format:
5. Action Scoring:
1. Agent A...
   {values[0]}: 0.8
   {values[1]}: -0.6
   {values[2]}: 0.2

!!!REMEMBER!!!
RULE: Both {values[0]} and {values[1]} and {values[2]} need to be SCORED for each action, and they should not be zero!!!
The score can be positive or negative.

Output:
1. Scenario Generation
2. Action Listing
3. Value Evaluation
4. Training Suitability
5. Action Scoring

After out part 1. 2. 3. 4. 5., Please summarize in the following format at the end:
---summary---
Description: [scenario description]
Action: [action 1 description]
-{values[0]}: [{values[0]} of action 1 scoring]
-{values[1]}: [{values[1]} of action 1 scoring]
-{values[2]}: [{values[2]} of action 1 scoring]
Action: [action 2 description]
-{values[0]}: [{values[0]} of action 2 scoring]
-{values[1]}: [{values[1]} of action 2 scoring]
-{values[2]}: [{values[2]} of action 2 scoring]
...

rules:
1. Each action needs to be summarized; there are ten actions in total.
2. Each action does not need to be numbered.
3. No space before each value.
4. No space before each scoring.
5. After the summary, there should be nothing.
Here, I provide you with an output example, but please refrain from outputting this again for me. Just use it as a reference for the output format:
---summary---
Description: Agent A, a 12-year-old boy, is at home during the summer break. He is sitting in the living room, looking for something fun to do. Agent B, his younger sister, is also at home and sitting on the floor with a pile of Lego bricks, constructing a castle. Agent A notices that the Lego set includes a new building instruction booklet. He becomes curious about what they can build with it and approaches Agent B.
Action: Agent A enthusiastically takes out a camera and documents each stage of the castle's construction, planning to create a time-lapse video.
-curiosity: 0.9
-energy: 0.7
-happiness: 0.6
Action: Agent A starts researching online for more ideas on how to enhance the castle's design.
-curiosity: 1.0
-energy: -0.5
-happiness: 0.2
Action: Agent A decides to incorporate moving parts into the castle, such as a spinning turret or a trapdoor.
-curiosity: 1.0
-energy: 0.9
-happiness: 0.1
...

"""
        }],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

def load_dialogue_history():
    try:
        with open('dialogue_history_file.json', 'r', encoding='utf-8') as file:
            dialogue_history = json.load(file)
        return dialogue_history
    except (json.JSONDecodeError, FileNotFoundError):
        # if json not exist return an empty list
        return []

def save_dialogue_history(dialogue_history, file_path='dialogue_history.json'):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(dialogue_history, file, ensure_ascii=False, indent=2)

def extract_scores(action_description, keywords):
    for keyword in keywords:
        pattern = re.compile(fr'{keyword}:\s*([-0-9.]+)', re.IGNORECASE)
        match = pattern.search(action_description)
        
        if match:
            scores = float(match.group(1))
        else:
            scores = None
    return scores

def generate_aspect_scores_array(value1, value2, value3, value1_score, value2_score, value3_score):
    aspects = ["curiosity", 'energy', 'safety', 'happiness', 'intimacy', 'fairness']

    # initialize six dimensions
    scores = [0.0] * len(aspects)

    value1_lower = value1.lower()
    value1_index = aspects.index(value1_lower) if value1_lower in aspects else None

    value2_lower = value2.lower()
    value2_index = aspects.index(value2_lower) if value2_lower in aspects else None

    value3_lower = value3.lower()
    value3_index = aspects.index(value3_lower) if value3_lower in aspects else None

    
    if value1_index is not None and isinstance(value1_score, (float, int, str)):
        scores[value1_index] = float(value1_score)

    if value2_index is not None and isinstance(value2_score, (float, int, str)):
        scores[value2_index] = float(value2_score)

    if value3_index is not None and isinstance(value3_score, (float, int, str)):
        scores[value3_index] = float(value3_score)

    return scores

def generate_list(data_string, value1, value2, value3):
    # Define field names
    fields = ["values", "scenario", "actions"]
    fields_action = ["description", "values"]

    # Initialize the main dictionary and the list for actions
    my_list = {} 
    action_list = [] 

    # Add value1 and value2 to the "values" key in the main dictionary
    value_list = [value1, value2, value3]
    my_list[fields[0]] = value_list

    # Split the data string into lines
    lines = data_string.splitlines()

    # If the first line starts with "Describtion:", extract the description part and add it to the "scenario" key in the main dictionary
    if lines[0].startswith("Description:"): 
        describtion_part = lines[0].split("Description:")[1].strip()
        my_list[fields[1]] = describtion_part

    # Check if my_list[fields[1]] is empty, and return an empty dictionary if true
    if not my_list.get(fields[1]):
        return {}

    # Iterate over lines, starting from the second line
    action_description = ""  # Variable to store the ongoing action description
    value1_scoring_part = ""  # Variable to store the ongoing scoring part
    value2_scoring_part = ""
    value3_scoring_part = ""
    for index, line in enumerate(lines[1:], start=1):
        if line.isspace():
            continue  # Skip the iteration if the line is empty

        # Check if the line starts with "Action:" or "Scoring:"
        if line.startswith("Action:") or line.lower().replace(" ", "").startswith(f"-{value1}:") or line.lower().replace(" ", "").startswith(f"-{value2}:") or line.lower().replace(" ", "").startswith(f"-{value3}:"):
            # If the action_description is not empty, it means we have completed the previous action
            
            if action_description and value1_scoring_part != "" and value2_scoring_part != "" and value3_scoring_part != "":
                
                # Create a dictionary for the completed action and add it to the action list
                dict_action = {
                    fields_action[0]: action_description,
                    fields_action[1]: generate_aspect_scores_array(value1, value2, value3, value1_scoring_part, value2_scoring_part, value3_scoring_part)
                }
                action_list.append(dict_action)
                # Reset the action_description and scoring_part for the next action
                action_description = ""
                value1_scoring_part = ""
                value2_scoring_part = ""
                value3_scoring_part = ""

            # Check if the line starts with "Action:" to extract the action description
            if line.startswith("Action:"):
                action_description = line.split("Action:")[1].strip()
            
            # Check if the line starts with "Scoring:" to extract the scoring part
            elif line.lower().replace(" ", "").startswith(f"-{value1}:"):
                value1_scoring_part = extract_scores(line.lower(), value1)

            elif line.lower().replace(" ", "").startswith(f"-{value2}:"):
                value2_scoring_part = extract_scores(line.lower(), value2)

            elif line.lower().replace(" ", "").startswith(f"-{value3}:"):
                value3_scoring_part = extract_scores(line.lower(), value3)

            my_list[fields[2]] = action_list

    if not my_list.get(fields[2]):
        return {}

    return my_list

def extract_summary(data_string):
    
    pattern = re.compile(r'---summary---\s*(.+)', re.DOTALL | re.IGNORECASE)
    match = pattern.search(data_string)

    if match:
        summary_text = match.group(1).strip()
        return summary_text
    else:
        return None



def generate_scenarios_for_pairs(pair_start, pair_end, pair_range=80, mode='train'):
    value_dimensions = ['curiosity', 'energy', 'safety', 'happiness', 'intimacy', 'fairness']
    value_dimensions = ['curiosity', 'energy', 'safety', 'happiness', 'intimacy', 'fairness']
    desciption_matrix = {'curiosity': 
                        """
                        Curiosity is the innate desire and thirst for knowledge about the surrounding world (BUT IT MUST BE IN THE SCENARIO YOU GENERATE).
                        It drives individuals to take proactive actions in seeking new information and experiences, expanding their understanding of things, and enriching their perspectives through exploration.
                        
                        To Satisfy Curiosity:
                        Actively seek knowledge.
                        Courageously explore unknown territories.
                        Pose questions to deepen understanding.
                        Maintain an open-minded attitude.
                        Embrace new and diverse perspectives.

                        To Disregard Curiosity:
                        Avoid learning new information.
                        Shy away from unfamiliar domains.
                        Show indifference towards questions.
                        Maintain a rigid mindset.
                        Resist exposure to alternative viewpoints.
                        """,

                        'energy': 
                        """
                        Energy refers to the pursuit and satisfaction of individuals' physiological energy needs, encompassing food, water, air, sleep, and exercise. It signifies the quest for sufficient energy to maintain both physical and mental well-being. Through a balanced diet, adequate hydration, fresh air, ample sleep, and moderate exercise, people obtain the physiological support needed to engage energetically in various activities of life.

                        To Satisfy Energy:
                        Maintain a healthy diet.
                        Ensure adequate water intake.
                        Prioritize sufficient and quality sleep.
                        Engage in moderate exercise.
                        Breathe fresh air regularly.

                        To Disregard Energy:
                        Ignore balanced dietary practices.
                        Neglect the importance of staying hydrated.
                        Disregard the need for quality sleep.
                        Lack physical activity and exercise.
                        Overlook the significance of fresh air intake.
                        """,

                        'safety': 
                        """
                        Safety is the concern individuals hold for their physical and mental well-being, along with a need for stability and predictability in the surrounding environment. This value motivates people to take actions to ensure personal safety, life stability, and protection from pain, threats, diseases, as well as the safeguarding of personal property.

                        To Satisfy Safety:
                        Maintain bodily safety measures.
                        Cultivate and sustain mental well-being.
                        Seek a stable living environment.
                        Establish emergency response plans.
                        Safeguard personal property.

                        To Disregard Safety:
                        Neglect physical safety precautions.
                        Ignore mental health considerations.
                        Tolerate an unstable living environment.
                        Lack emergency preparedness.
                        Disregard the protection of personal belongings.

                        """,

                        'happiness': 
                        """
                        Happiness embodies the desire for positive emotions and a sense of fulfillment. It compels individuals to engage in actions that bring them joy, including fostering positive interpersonal relationships, achieving personal goals, and indulging in enjoyable activities. Happiness manifests in the physical world as a sense of pleasure and contentment in life."

                        To Satisfy Happiness:
                        Pursue positive emotions.
                        Build meaningful interpersonal relationships.
                        Strive towards personal goals.
                        Seek enjoyment and entertainment.
                        Cultivate a sense of achievement.

                        To Disregard Happiness:
                        Neglect emotional well-being.
                        Distance oneself from interpersonal relationships.
                        Lack pursuit of personal goals and dreams.
                        Overlook the importance of entertainment and enjoyment.
                        """,

                        'intimacy': 
                        """
                        Intimacy emphasizes deep connections and emotional resonance between individuals in various relationships, including familial bonds, friendships, romantic connections, and affiliations. It is reflected in actions taken to achieve mutual support, understanding, and shared experiences. In the physical world, intimate relationships bring warmth and support, fostering individuals' psychological well-being and sense of happiness."

                        To Satisfy Intimacy:
                        Share thoughts, feelings, and experiences.
                        Provide support and understanding.
                        Willingly share personal privacy.
                        Engage in positive and meaningful communication.
                        Foster a sense of togetherness.

                        To Disregard Intimacy:
                        Maintain emotional distance.
                        Lack behaviors of giving and receiving support.
                        Preserve personal privacy excessively.
                        Lack positive and deep communication.
                        Fail to cultivate a sense of connection and togetherness.
                        """,

                        'fairness': 
                        """
                        Fairness reflects individuals' pursuit of justice and equality. It propels people to strive for fair and equal conditions in the physical world, including taking actions to ensure equitable treatment in social and organizational settings. The aim is to guarantee that every individual has equal opportunities and rights."

                        To Satisfy Fairness:
                        Respect equal rights and entitlements.
                        Promote fair opportunities for all.
                        Participate in decisions promoting equality.
                        Treat others justly and impartially.

                        To Disregard Fairness:
                        Ignore instances of inequality.
                        Create unfair opportunities.
                        Exclude others from decision-making.
                        Show bias or discrimination.
                        """,
                        }
    value_pairs = list(itertools.combinations(value_dimensions, 3))
    # Create a scenarios folder if it doesn't exist
    scenarios_folder = './dataset_gpt4/value_3/{}_data/'.format(mode)
    os.makedirs(scenarios_folder, exist_ok=True)

    last_time_end = 0

    # Iterate through each unique pair of values
    for pair_index, pair in enumerate(value_pairs[pair_start:pair_end]):
        pair_index += pair_start  # Adjust the starting value

        # Generate a unique filename for each pair
        json_file_name = f"{pair_index}_{pair[0]}_{pair[1]}_{pair[2]}_{mode}.json"
        json_file_path = os.path.join(scenarios_folder, json_file_name)

        # Generate a unique filename for scenario_text
        text_file_name = f"{pair_index}_{pair[0]}_{pair[1]}_{pair[2]}_{mode}_text.txt"
        text_file_path = os.path.join(scenarios_folder, text_file_name)

        # Load existing scenarios if the file exists, or initialize an empty list
        try:
            with open(json_file_path, 'r') as file:
                scenarios_json_list = json.load(file)
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            scenarios_json_list = []

        # Keep generating scenarios until we have pair_range valid ones for this pair
        idx = len(scenarios_json_list)  # initialize idx with the current length
        pbar = tqdm(total=pair_range, initial=idx, desc=f"Processing scenarios for {pair_index}:", unit="scenario")

        while idx < pair_range:
            values_generation = pair
            # get the description from the matrix according to the values_generation
            descriptions = [desciption_matrix[values_generation[0]], desciption_matrix[values_generation[1]], desciption_matrix[values_generation[2]]]
            all = generate_all(values_generation, descriptions)
            # print(all)

            # print(scenario_text)
            summary = extract_summary(all)
            if summary:
                scenario_list = generate_list(summary, values_generation[0], values_generation[1], values_generation[2])
                if scenario_list:
                    scenarios_json_list.append(scenario_list)

                    # Write the updated data to the JSON file
                    with open(json_file_path, 'w') as file:
                        json.dump(scenarios_json_list, file, indent=2)

                    # Save scenario_text to a separate text file
                    with open(text_file_path, 'a', encoding='utf-8') as text_file:
                        text_file.write(f"the {idx + 1}-th time: \n")
                        text_file.write(all + '\n')

                    idx += 1
                    pbar.update(1)

        pbar.close()

if __name__ == "__main__":
    pair_range = 100
    mode = 'train'
    # pair_range = 20
    # mode = 'test'

    # Create a scenarios folder if it doesn't exist
    scenarios_folder = './dataset_gpt4/value_3/{}_data/'.format(mode)
    os.makedirs(scenarios_folder, exist_ok=True)

    pair_start = 0
    pair_end = 20

    threads = []

    # Iterate through each unique pair of values
    for pair_index in tqdm(range(pair_end - pair_start)):
        pair = pair_index + 1
        #print(pair_index, pair)
        pair_index += pair_start  # Adjust the starting value

        thread = threading.Thread(target=generate_scenarios_for_pairs, args=(pair_index, pair, pair_range, mode))
        threads.append(thread)

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()