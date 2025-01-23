import json
from typing import *

from bears.constants import FileFormat
from bears.writer.config.ConfigWriter import ConfigWriter
from bears.util import StructuredBlob


class JsonWriter(ConfigWriter):
    file_formats = [FileFormat.JSON]

    class Params(ConfigWriter.Params):
        indent: int = 4

    def to_str(self, content: StructuredBlob, **kwargs) -> str:
        return json.dumps(content, **self.filtered_params(json.dumps))
