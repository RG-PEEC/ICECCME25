import pandas as pd
import ast

class VolunteerDataset:
    def __init__(self):
        self.df = pd.read_csv(
            "/Users/p42939_christophgassner/Code/iceccme_experiments/datasets/volunteer/volunteer_skills_mapped.csv",
            sep=";", converters={"y": ast.literal_eval},)
        self.df_y = pd.read_csv(
            "/Users/p42939_christophgassner/Code/iceccme_experiments/datasets/volunteerSoft/volunteer_softskills_16_05.csv",
            sep=";", converters={"y": ast.literal_eval})
        self.df_merge = pd.merge(self.df_y, self.df, how="left", on="task_id")
        # merge the lists in y_x and y_y to a single list in y
        self.df_merge["y"] = self.df_merge["y_x"]
        # strip all the strings in the list in y
        self.df_merge["y"] = self.df_merge["y"].apply(lambda x: [i.strip().rstrip() for i in x])
        # remove \n from the strings in the list in y
        self.df_merge["y"] = self.df_merge["y"].apply(lambda x: [i.replace("\n", "") for i in x])
        # self.df_merge["y"] = self.df_merge["y"].apply(lambda x: [i.replace("SW", "software") for i in x])
        # self.df_merge["y"] = self.df_merge["y"].apply(lambda x: [i.replace("HW", "hardware") for i in x])


    def get_data(self) -> ([str], [[str]]):
        """
        Get the data from the dataset.
        :return: A tuple of two lists: the input data and the evaluation data.
        """
        input_data = self.df["description"].tolist()
        # evaluation_data = self.df_y["y"].tolist()
        evaluation_data = self.df_merge["y"].tolist()
        return input_data, evaluation_data

    def get_few_shot_examples(self, n=1) -> [[str]]:
        """
        Get the few shot examples from the dataset.
        :return: A list of few shot examples.
        """
        if n == 1:
            few_shot_examples = [
                {
                    "description": "Are you a data geek and do you want to be part of the digitalisation journey at a company working towards ensuring a world that runs entirely on green energy? Join us and become Senior Data Engineer in Advanced Analytics where you¬íll be part of projects and labs responsible for developing our new data backbone and inspiring the stakeholders in advanced analytics and artificial intelligence techniques to achieve better results. The Advanced Analytics team consists of 20 professionals within the area of Big Data Engineering and Data Science.",
                    "output": "['carry out calculations', 'calculate probabilities', 'interpret mathematical information', 'operate digital hardware', 'use communication & collaboration software', 'conduct web searches', 'apply basic programming skills', 'think analytically', 'organise information', 'objects & resources', 'identify problems', 'solve problems', 'work in teams', 'critically evaluate information and its sources', 'apply knowledge of science, technology & engineering']"
                },
                {
                    "description": "Volunteer at the St. Francis Living Room to help homeless & low income seniors with their breakfast!\nAt the St. Francis Living Room, we serve a nutritious breakfast to low income and homeless seniors in the Tenderloin from Monday to Friday. Everyone is treated with dignity, kindness and respect. Volunteers welcome guests, help serve meals, assist those seniors needing extra assistance and help clean up. This is an enrichening experience for anyone.While ongoing volunteers are preferable, one time helpers are also welcome. We want to encourage San Francisco visitors to join us for a morning during their travel. We are open weekday mornings and volunteer shifts typically start at 8 a.m. till 10 a.m. The St. Francis Living Room is located near Civic Center, easily accessible via Muni or BART. Come spend a morning or two with us and feel how rewarding it can be to help those who need it the most. For more information, please contact Pierre Smit at Pierre@SFLivingRoom.org or leave a voicemail at 415-946-1413. Or visit our website at http://sflivingroom.org. We're on Facebook (https://www.facebook.com/groups/1829004774143524)",
                    "output": "['work efficiently', 'address an audience', 'ensure customer orientation', 'show empathy', 'move objects', 'protect the health of others', 'demonstrate awareness of health risks', 'apply hygiene standards']"
                }
            ]
        else:
            few_shot_examples = []
        return few_shot_examples


volunteerDataset = VolunteerDataset()
"""x, y = volunteerDataset.get_data()
for task, list in zip(x, y):
    print(task)
    print(list)
    print()"""

