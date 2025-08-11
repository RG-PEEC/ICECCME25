from langchain.tools import tool
from utils.esco import EscoSkillFinder
import requests

def extract_skills_from_response(response):
    """Extract skills from the ESCO API response."""
    skills = []
    for item in response.get("_embedded").get("results", []):
        skill_label = item.get("preferredLabel").get("en")
        # skill_link = item.get("_links").get("self").get("href")
        skill_uri = item.get("_links").get("self").get("uri")
        skill_description = EscoSkillFinder().get_description_by_uri(skill_uri)

        skill = {
            "preferredLabel": skill_label,
            "description": skill_description,
            # "link": skill_link,
            "uri": skill_uri,
            "notFoundInESCO": skill_description is None,
        }
        skills.append(skill)
    return skills

@tool
def search_esco_api(query: str, limit: int = 20) -> [dict]:
    """Search for ESCO skills using the ESCO API.
    Args:
        query (str): The search query.
        limit (int): The maximum number of results to return.
    Returns:
        str: The response from the ESCO API."""
    url = "http://localhost:8080/search"
    headers = {
        "Accept": "application/json,application/json;charset=UTF-8"
    }
    params = {
        "text": query,
        "language": "en",
        "limit": limit,
        "type": "skill"
    }

    response = requests.get(url, headers=headers, params=params)

    # print get url
    # print(response.url)
    response.raise_for_status()  # Fehler werfen bei HTTP-Fehlern
    return extract_skills_from_response(response.json())