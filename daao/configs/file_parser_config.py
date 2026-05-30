from daao.utils.yaml_model import YamlModel


class OmniParseConfig(YamlModel):
    api_key: str = ""
    base_url: str = ""
