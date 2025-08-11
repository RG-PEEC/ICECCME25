import pandas as pd


class VolunteerDatasetSoft:
    def __init__(self):
        self.df = pd.read_csv("/Users/p42939_christophgassner/Code/iceccme_experiments/datasets/volunteerSoft/final_sample.csv", sep=";")
        self.df_y = pd.read_csv("/Users/p42939_christophgassner/Code/iceccme_experiments/datasets/volunteerSoft/volunteer_softskills.csv", sep=";")

    def get_data(self) -> ([str], [[str]]):
        """
        Get the data from the dataset.
        :return: A tuple of two lists: the input data and the evaluation data.
        """
        input_data = self.df["description"].tolist()
        evaluation_data = self.df_y["y"].tolist()
        return input_data, evaluation_data

    def get_few_shot_examples(self, n = 1) -> [[str]]:
        """
        Get the few shot examples from the dataset.
        :return: A list of few shot examples.
        """
        if n == 1:
            few_shot_examples = [
                {
                    "description": "Are you a data geek and do you want to be part of the digitalisation journey at a company working towards ensuring a world that runs entirely on green energy? Join us and become Senior Data Engineer in Advanced Analytics where you¬íll be part of projects and labs responsible for developing our new data backbone and inspiring the stakeholders in advanced analytics and artificial intelligence techniques to achieve better results . The Advanced Analytics team consists of 20 professionals within the area of Big Data Engineering and Data Science . <ORGANIZATION> <ORGANIZATION> <ORGANIZATION> are looking for new staff to housekeeping . Working place is in <LOCATION> The job is daily cleaning of hotelrum Workingtime: from monday to friday and also in week-ends and it will be about 20/30 hours pr . week . There will be prepared a schedule for every month",
                    "output": "['principles of artificial intelligence', 'engage with stakeholders', 'develop data processing applications']"
                }
            ]
        else:
            few_shot_examples = []
        return few_shot_examples

volunteerDatasetSoft = VolunteerDatasetSoft()
