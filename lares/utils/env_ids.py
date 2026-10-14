"""The one place the project's ``-v2`` task names are mapped to MetaWorld 3 ids."""

ENV_ID_MAPPING_RULE = "metaworld>=3.0: replace '-v2' (or '-v1') with '-v3' before looking up ALL_V3_ENVIRONMENTS"
ENV_ID_MAPPING_VERSION = "v2-to-v3/1"


def resolve_registry_env_id(env_id: str) -> str:
    """The id MetaWorld 3 registers for a project-facing ``-v2`` task name."""
    return str(env_id).replace("-v2", "-v3").replace("-v1", "-v3")
