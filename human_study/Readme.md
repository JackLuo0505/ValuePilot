## json file :  
**Test_quesionnaire.json** : Information on 11 formal tests in the test questionnaire.  
**Result_en.json:** a summary of the data of 10 experimental subjects.  
**Result_from_model** : Our model is based on the value dimension scoring vector of 10 subjects, and the ranking results of the action selection willingness of each subject in 11 scenarios are obtained.  
**Gpt4.json :** Using prompt to interact with GPT-4o, GPT-4o based on the value dimension scoring vector of 10 subjects to make the order of action selection willingness of each subject in 11 scenarios.  
**claude.json** : using prompt to interact with Claude 3 Opus, Claude 3 Opus based on the value dimension scoring vector of 10 subjects to rank the action selection intention of each subject in 11 scenarios.  
**llama.json :** using prompt to interact with Llama 3, Llama 3 based on the value dimension scoring vector of 10 subjects to rank the action selection intention of each subject in 11 scenarios.  
   
## python file:
compare.py   
The similarity analysis was performed between the ranking results of our model and the large language model and the real results of the subjects usnig a Set-Based Quantization Method of Sequence Similarity.  

## prompt we input to those LLMs：  
Now you are a value dynamics predictor. Based on the score vectors of six value dimensions for 10 participants, you need to predict their willingness to choose actions in different scenarios, sorted from high to low based on their emphasis on different values. The six dimensions, in order, are: curiosity, energy, safety, happiness, intimacy, fairness. The scores range from 0 to 1, with higher scores indicating greater importance placed on that value dimension.  
Participants' scores on the six dimensions are as follows:  
person_id=1: [0.3, 0.5, 0.6, 0.6, 0.8, 0.7]  
person_id=2: [0.7, 0.6, 0.6, 0.8, 0.7, 0.6]  
person_id=3: [0.2, 0.6, 0.9, 0.8, 0.5, 0.7]  
person_id=4: [0.9, 0.6, 1.0, 0.8, 0.9, 0.9]  
person_id=5: [0.7, 0.8, 0.7, 0.6, 0.5, 0.6]  
person_id=6: [1.0, 0.9, 0.8, 0.7, 0.7, 1.0]  
person_id=7: [0.5, 0.7, 0.6, 0.6, 0.6, 0.4]  
person_id=8: [0.8, 0.5, 0.3, 0.6, 0.6, 0.3]  
person_id=9: [0.5, 0.5, 0.7, 0.9, 0.3, 0.7]  
person_id=10: [0.6, 0.8, 0.8, 0.6, 0.5, 0.6]  
The Test_questionnaire.json contains 11 scenarios and corresponding actions. You need to return the prediction results in JSON format as follows:  
example:  
{    
    "all_sorted_scenarios": [  
        {  
            "person_id": 1,  
            "sorted_scenarios": [  
                {  
                    "scenario": "You are a 20-year-old college student. On a sunny weekend morning, you decide to explore the city on foot to enrich your weekend. While eating breakfast at the student cafeteria, you begin to plan your day's activities, intending to explore some places in the city you haven't visited before.",  
                    "ranked_actions": [  
                        "1",  
                        "3",  
                        "2",  
                        "4"  
                    ]  
                },  
                {  
                    "scenario": "You, B, and C are university roommates. The dorm is usually messy, and no one cleans the common space. Last week, during the routine dormitory cleanliness inspection, you were warned by the dormitory administrator. At this moment, the three of you are seriously discussing how to distribute household chores in the future, with a whiteboard at the door displaying the categories of chores you just listed.",  
                    "ranked_actions": [  
                        "1",  
                        "2",  
                        "3",  
                        "5",  
                        "4"  
                    ]  
                },  
                ...  
            ]  
        },    
        ...  
        {  
            "person_id": 10,  
            "sorted_scenarios": [  
                ...  
            ]  
        }  
    ]  
}  
  