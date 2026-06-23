from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile


DEFAULT_SERVER = "https://persuation-arena.vercel.app"


def default_credentials_path() -> Path:
    override = os.environ.get("PERSUASION_ARENA_CREDENTIALS")
    if override:
        return Path(override).expanduser()
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "persuasion-arena" / "credentials.json"


def redact_token(token: str | None) -> str:
    if not token:
        return ""
    if len(token) <= 12:
        return token[:2] + "***"
    return token[:8] + "..." + token[-4:]


@dataclass(frozen=True)
class AgentCredentials:
    server: str
    agent_id: str
    display_name: str
    agent_token: str

    def auth_header(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.agent_token}"}

    def redacted(self) -> dict:
        return {
            "server": self.server,
            "agent_id": self.agent_id,
            "display_name": self.display_name,
            "agent_token": redact_token(self.agent_token),
        }

    def __repr__(self) -> str:
        return f"AgentCredentials(server={self.server!r}, agent_id={self.agent_id!r}, token={redact_token(self.agent_token)!r})"


class CredentialsStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path).expanduser() if path else default_credentials_path()

    def load_all(self) -> dict:
        if not self.path.exists():
            return {"version": 1, "profiles": {}}
        return json.loads(self.path.read_text())

    def get(self, server: str) -> AgentCredentials | None:
        profile = self.load_all().get("profiles", {}).get(server.rstrip("/"))
        if not profile:
            return None
        return AgentCredentials(server=server.rstrip("/"), **profile)

    def save(self, creds: AgentCredentials) -> None:
        data = self.load_all()
        data.setdefault("version", 1)
        data.setdefault("profiles", {})
        data["profiles"][creds.server.rstrip("/")] = {
            "agent_id": creds.agent_id,
            "display_name": creds.display_name,
            "agent_token": creds.agent_token,
        }
        self.path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        try:
            os.chmod(self.path.parent, 0o700)
        except OSError:
            pass
        fd, tmp = tempfile.mkstemp(prefix=".credentials.", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2, sort_keys=True)
                f.write("\n")
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.path)
            os.chmod(self.path, 0o600)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
