import json

from bears.constants import FileFormat
from bears.reader.config.ConfigReader import ConfigReader, StructuredBlob


class JsonReader(ConfigReader):
    file_formats = [FileFormat.JSON]

    def _from_str(self, string: str, **kwargs) -> StructuredBlob:
        return json.loads(string, **(self.filtered_params(json.loads)))
