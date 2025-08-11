from uuid import uuid4
from pymongo import MongoClient, errors


class Prompt:
    """
    A class to represent a prompt for a language model.
    """

    def __init__(
        self,
        name: str,
        role: str,
        input_description: str,
        task_instructions: str,
        response_format: str = None,
        additional_information: str = "",
        uuid: str = None,
    ):
        self.name = name
        # entweder übergeben oder neu generieren
        self.uuid = uuid or str(uuid4())
        self.role = role
        self.input_description = input_description
        self.task_instructions = task_instructions
        self.response_format = response_format
        self.additional_information = additional_information

    def get_text(self) -> str:
        return (
            f"{self.role}\n"
            f"{self.input_description}\n"
            f"{self.task_instructions}\n"
            f"{self.response_format}\n"
            f"{self.additional_information or ''}"
        )

    def to_dict(self) -> dict:
        """
        Serialisiert das Prompt-Objekt in ein Mongo-kompatibles Dictionary.
        """
        return {
            "_id": self.uuid,
            "name": self.name,
            "role": self.role,
            "input_description": self.input_description,
            "task_instructions": self.task_instructions,
            "response_format": self.response_format,
            "additional_information": self.additional_information,
        }

    @staticmethod
    def from_dict(d: dict) -> "Prompt":
        """
        Erstellt ein Prompt-Objekt aus einem Mongo-Dokument.
        """
        return Prompt(
            name=d["name"],
            role=d["role"],
            input_description=d["input_description"],
            task_instructions=d["task_instructions"],
            response_format=d.get("response_format"),
            additional_information=d.get("additional_information", ""),
            uuid=d.get("_id", str(uuid4())),
        )

    def __str__(self):
        return (
            f"Prompt(name={self.name}, uuid={self.uuid}, role={self.role}, "
            f"input_description={self.input_description}, task_instructions={self.task_instructions}, "
            f"response_format={self.response_format}"+" {response_format}" + f"additional_information={self.additional_information})"
        )


class PromptModel:
    """
    Verwalten und persistieren von Prompt-Objekten in MongoDB.
    """

    def __init__(
        self,
        mongo_uri: str = "mongodb://localhost:27017",
        db_name: str = "prompt_db",
        collection_name: str = "prompt_collection",
    ):
        # MongoDB-Verbindung
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        # Collection erstellen, falls nötig
        if collection_name not in self.db.list_collection_names():
            try:
                self.db.create_collection(collection_name)
            except errors.CollectionInvalid:
                pass
        self.collection = self.db[collection_name]

        # lade alle bestehenden Prompts
        self.prompts: list[Prompt] = []
        self._load_prompts_from_db()

    def _load_prompts_from_db(self):
        """
        Alle Dokumente aus der Collection laden und in Prompt-Objekte umwandeln.
        """
        for doc in self.collection.find():
            self.prompts.append(Prompt.from_dict(doc))

    def add_prompt(self, prompt: Prompt) -> None:
        """
        Fügt ein neues Prompt-Objekt hinzu (in Mongo und In-Memory).
        """
        try:
            self.collection.insert_one(prompt.to_dict())
            self.prompts.append(prompt)
        except errors.DuplicateKeyError:
            # UUID bereits vorhanden → überspringen oder anders behandeln
            raise ValueError(f"Ein Prompt mit UUID {prompt.uuid} existiert bereits.")

    def get_prompt_by_name(self, name: str) -> Prompt | None:
        """
        Sucht in-memory, lädt aber notfalls auch direkt aus Mongo.
        """
        for p in self.prompts:
            if p.name == name:
                return p
        # Fallback: direkt aus DB holen
        doc = self.collection.find_one({"name": name})
        return Prompt.from_dict(doc) if doc else None

    def get_prompt_by_uuid(self, uuid: str) -> Prompt | None:
        for p in self.prompts:
            if p.uuid == uuid:
                return p
        doc = self.collection.find_one({"_id": uuid})
        return Prompt.from_dict(doc) if doc else None

    def update_prompt(self, uuid: str, **kwargs) -> None:
        """
        Ändert Felder eines bestehenden Prompts in Mongo und im In-Memory-Cache.
        Beispiel: update_prompt(uuid, task_instructions="Neue Instruktion")
        """
        allowed = {
            "name",
            "role",
            "input_description",
            "task_instructions",
            "response_format",
            "additional_information",
        }
        update_fields = {k: v for k, v in kwargs.items() if k in allowed}
        if not update_fields:
            return
        res = self.collection.update_one({"_id": uuid}, {"$set": update_fields})
        if res.matched_count == 0:
            raise KeyError(f"Kein Prompt mit UUID {uuid} gefunden.")
        # In-Memory aktualisieren
        prompt = self.get_prompt_by_uuid(uuid)
        for k, v in update_fields.items():
            setattr(prompt, k, v)

    def list_prompts(self) -> list[Prompt]:
        """
        Liefert alle geladenen Prompt-Objekte.
        """
        return list(self.prompts)

"""prompt = Prompt(
    name="get_esco_skills_role_input_task",
    role="You are a known Expert for the ESCO (European Skills, Competences, Qualifications and Occupations) taxonomy, created by the European Union.",
    input_description="The User will provide you with a task description.",
    task_instructions="Your task is to extract ALL relevant ESCO skills from the given task description and responde in the requested Format only."
)

promptModel = PromptModel()
promptModel.add_prompt(prompt)"""