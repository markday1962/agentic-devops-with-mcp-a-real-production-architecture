"""Entrypoint: ``uvicorn agentic_devops.main:app``.

Server configuration comes from the environment so the same image runs in
every cluster; the MCP server list is a mounted JSON file because it changes
independently of the code.
"""

from __future__ import annotations

import os

from .memory.fts import FTSKnowledgeIndex
from .memory.transcripts import TranscriptStore
from .observability.setup import configure_observability
from .orchestration.agent import AgentConfig, load_servers
from .triggers.app import TriggerSettings, create_app

# Configures logging too, so everything below emits trace-correlated records.
observability = configure_observability()

servers = load_servers(os.environ.get("MCP_SERVERS_FILE", "/etc/agent/servers.json"))
config = AgentConfig.from_env(
    servers,
    knowledge=FTSKnowledgeIndex(os.environ.get("KNOWLEDGE_DB", "/data/knowledge.db")),
    transcripts=TranscriptStore(os.environ.get("TRANSCRIPTS_DB", "/data/transcripts.db")),
)

app = create_app(TriggerSettings.from_env(config), observability=observability)
