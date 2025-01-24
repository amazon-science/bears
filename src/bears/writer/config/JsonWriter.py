import json

from bears.constants import FileFormat
from bears.util import StructuredBlob
from bears.writer.config.ConfigWriter import ConfigWriter


class JsonWriter(ConfigWriter):
    file_formats = [FileFormat.JSON]

    class Params(ConfigWriter.Params):
        indent: int = 4

    def to_str(self, content: StructuredBlob, **kwargs) -> str:
        return json.dumps(content, **self.filtered_params(json.dumps))
