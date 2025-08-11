import pandas as pd

def parse_alt_labels(alt_labels: str) -> list:
    """
    Parse the alternative labels from a string to a list.
    :param alt_labels: String of alternative labels
    :return: List of alternative labels
    """
    if pd.isna(alt_labels):
        return []
    alt_label_list = alt_labels.split('\n')
    return [label.strip() for label in alt_label_list if label is not None]



class EscoSkillFinder:
    def __init__(self):
        self.esco_df = pd.read_csv("/Users/p42939_christophgassner/Code/iceccme_experiments/data/transversalSkillsCollection_en.csv",
                                   converters={'altLabels': parse_alt_labels})
        self.esco_df['preferredLabel'] = self.esco_df['preferredLabel'].str.lower()
        self.esco_df['preferredLabel'] = self.esco_df['preferredLabel'].str.strip()
        self.esco_df['preferredLabel'] = self.esco_df['preferredLabel'].str.rstrip()
        self.esco_df['preferredLabel'] = self.esco_df['preferredLabel'].str.replace("\n", "")
        self.esco_df['altLabels'] = self.esco_df['altLabels'].apply(lambda x: [label.lower() for label in x])

    def get_skill_by_uri(self, uri: str):
        """
        Get skill by URI
        :param uri: URI of the skill
        :return: Skill name
        """
        skill = self.esco_df[self.esco_df['conceptUri'] == uri]
        if skill.empty:
            return None
        return skill.iloc[0]['preferredLabel']

    def check_label_in_esco_preferredLabel(self, preferredLabel: str) -> bool:
        """
        Get skill by preferredLabel
        :param preferredLabel: Preferred label of the skill
        :return: bool indicating if the skill exists in esco
        """
        preferredLabel = preferredLabel.lower()
        preferredLabel = preferredLabel.replace("HW", "hardware")
        preferredLabel = preferredLabel.replace("SW", "software")
        preferredLabel = preferredLabel.strip()
        preferredLabel = preferredLabel.rstrip()
        preferredLabel = preferredLabel.replace("\n", "")
        preferredLabel = preferredLabel.replace("&", "and")
        skill = self.esco_df[self.esco_df['preferredLabel'] == preferredLabel]
        if skill.empty:
            return False
        return True

    def check_label_in_esco_preferredLabel_or_altLabel(self, preferredLabel: str) -> bool:
        """
        Get skill by preferredLabel or altLabel
        :param preferredLabel: Preferred label of the skill
        :return: bool indicating if the skill exists in esco
        """
        preferredLabel = preferredLabel.lower().strip()
        skill = self.esco_df[
            (self.esco_df['preferredLabel'] == preferredLabel) |
            (self.esco_df['altLabels'].apply(lambda x: preferredLabel in x))
        ]
        if skill.empty:
            return False
        return True

    def get_description_by_uri(self, uri: str):
        """
        Get description by URI
        :param uri: URI of the skill
        :return: Skill description
        """
        skill = self.esco_df[self.esco_df['conceptUri'] == uri]
        if skill.empty:
            return None
        return skill.iloc[0]['description']

    def get_skill_uri_by_preferredLabel(self, preferredLabel: str):
        """
        Get skill URI by preferredLabel
        :param preferredLabel: Preferred label of the skill
        :return: Skill URI
        """
        skill = self.esco_df[self.esco_df['preferredLabel'] == preferredLabel]
        if skill.empty:
            return "Not Found"
        return skill.iloc[0]['conceptUri']

    def distance(self, skill1: str, skill2: str) -> float:
        """
        Calculate the distance between two skills
        :param skill1: First skill
        :param skill2: Second skill
        :return: Distance between the two skills
        """
        # Placeholder for actual distance calculation
        return -1
