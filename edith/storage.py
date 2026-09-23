from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import threading
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str
    timestamp: str


class GoogleDriveStorage:
    """Stores EDITH data as JSON files in the user's Google Drive."""

    def __init__(
        self,
        credentials_file: str,
        token_file: str,
        folder_name: str,
    ) -> None:
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.folder_name = folder_name
        self._service = None
        self._folders: dict[str, str] = {}
        self._lock = threading.RLock()

    def _api(self):
        if self._service is not None:
            return self._service
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
        except ImportError as error:
            raise RuntimeError(
                "Google Drive dependencies are missing. Install requirements.txt."
            ) from error

        scopes = ["https://www.googleapis.com/auth/drive.file"]
        credentials = None
        try:
            credentials = Credentials.from_authorized_user_file(self.token_file, scopes)
        except (FileNotFoundError, ValueError):
            pass
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        if not credentials or not credentials.valid:
            flow = InstalledAppFlow.from_client_secrets_file(
                self.credentials_file, scopes
            )
            credentials = flow.run_local_server(port=0)
        with open(self.token_file, "w", encoding="utf-8") as token:
            token.write(credentials.to_json())
        self._service = build("drive", "v3", credentials=credentials)
        return self._service

    def _folder_id(self, name: str, parent_id: str | None = None) -> str:
        key = f"{parent_id or 'root'}:{name}"
        if key in self._folders:
            return self._folders[key]
        query = (
            "mimeType = 'application/vnd.google-apps.folder' "
            f"and name = '{name.replace(chr(39), chr(39) + chr(39))}' "
            "and trashed = false"
        )
        if parent_id:
            query += f" and '{parent_id}' in parents"
        result = self._api().files().list(
            q=query, spaces="drive", fields="files(id, name)", pageSize=1
        ).execute()
        files = result.get("files", [])
        if files:
            folder_id = files[0]["id"]
        else:
            metadata: dict[str, Any] = {
                "name": name,
                "mimeType": "application/vnd.google-apps.folder",
            }
            if parent_id:
                metadata["parents"] = [parent_id]
            folder_id = self._api().files().create(
                body=metadata, fields="id"
            ).execute()["id"]
        self._folders[key] = folder_id
        return folder_id

    def _file_id(self, name: str, folder_id: str) -> str | None:
        safe_name = name.replace(chr(39), chr(39) + chr(39))
        result = self._api().files().list(
            q=f"name = '{safe_name}' and '{folder_id}' in parents and trashed = false",
            spaces="drive",
            fields="files(id, name)",
            pageSize=1,
        ).execute()
        files = result.get("files", [])
        return files[0]["id"] if files else None

    def _file_ids(self, folder_id: str) -> list[tuple[str, str]]:
        result = self._api().files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            spaces="drive",
            fields="files(id, name)",
            pageSize=100,
        ).execute()
        return [(item["id"], item["name"]) for item in result.get("files", [])]

    def read_json(self, collection: str, name: str, default: Any) -> Any:
        from googleapiclient.http import MediaIoBaseDownload
        import io

        with self._lock:
            folder_id = self._folder_id(collection, self._folder_id(self.folder_name))
            file_id = self._file_id(name, folder_id)
            if not file_id:
                return default
            request = self._api().files().get_media(fileId=file_id)
            buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(buffer, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            return json.loads(buffer.getvalue().decode("utf-8"))

    def write_json(self, collection: str, name: str, value: Any) -> None:
        from googleapiclient.http import MediaIoBaseUpload
        import io

        with self._lock:
            folder_id = self._folder_id(collection, self._folder_id(self.folder_name))
            payload = json.dumps(value, ensure_ascii=True, indent=2).encode("utf-8")
            media = MediaIoBaseUpload(io.BytesIO(payload), mimetype="application/json")
            file_id = self._file_id(name, folder_id)
            if file_id:
                self._api().files().update(fileId=file_id, media_body=media).execute()
            else:
                self._api().files().create(
                    body={"name": name, "parents": [folder_id]},
                    media_body=media,
                    fields="id",
                ).execute()

    def upload_binary(
        self, collection: str, name: str, path: str | Any, mime_type: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        from googleapiclient.http import MediaFileUpload

        with self._lock:
            folder_id = self._folder_id(collection, self._folder_id(self.folder_name))
            body: dict[str, Any] = {"name": name, "parents": [folder_id]}
            if metadata:
                body["description"] = "; ".join(f"{key}={value}" for key, value in metadata.items())
            result = self._api().files().create(
                body=body,
                media_body=MediaFileUpload(str(path), mimetype=mime_type, resumable=True),
                fields="id",
            ).execute()
            return result["id"]

    def delete_file(self, file_id: str) -> None:
        with self._lock:
            self._api().files().delete(fileId=file_id).execute()

    # Kept as a descriptive alias for callers handling uploaded media.
    def delete_binary(self, file_id: str) -> None:
        self.delete_file(file_id)

    def list_json(self, collection: str) -> list[tuple[str, Any]]:
        with self._lock:
            folder_id = self._folder_id(collection, self._folder_id(self.folder_name))
            records: list[tuple[str, Any]] = []
            for _, name in self._file_ids(folder_id):
                if name.endswith(".json"):
                    records.append((name, self.read_json(collection, name, {})))
            return records

    def append_message(self, conversation_id: str, role: str, content: str) -> None:
        conversation = self.read_json(
            "conversations",
            f"{conversation_id}.json",
            {"id": conversation_id, "messages": []},
        )
        conversation["messages"].append(
            asdict(
                ChatMessage(
                    role=role,
                    content=content,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            )
        )
        self.write_json("conversations", f"{conversation_id}.json", conversation)

    def active_conversation_id(self) -> str:
        metadata = self.read_json("metadata", "session.json", {})
        conversation_id = metadata.get("active_conversation_id")
        if not conversation_id:
            conversation_id = str(uuid4())
            self.write_json(
                "metadata", "session.json", {"active_conversation_id": conversation_id}
            )
        return conversation_id

    def remove_last_message(self, conversation_id: str) -> None:
        conversation = self.read_json(
            "conversations", f"{conversation_id}.json", {"id": conversation_id, "messages": []}
        )
        if conversation.get("messages"):
            conversation["messages"].pop()
            self.write_json("conversations", f"{conversation_id}.json", conversation)

    def recent_messages(self, conversation_id: str, limit: int = 12) -> list[dict[str, str]]:
        conversation = self.read_json(
            "conversations", f"{conversation_id}.json", {"messages": []}
        )
        return [
            {"role": item["role"], "content": item["content"]}
            for item in conversation.get("messages", [])[-limit:]
        ]

    def relevant_context(self, query: str, limit: int = 8) -> list[str]:
        memory = self.read_json("memory", "facts.json", {"facts": []})
        terms = {term.lower() for term in query.split() if len(term) > 2}
        scored = []
        for fact in memory.get("facts", []):
            text = str(fact.get("text", ""))
            score = sum(term in text.lower() for term in terms)
            scored.append((score, text))
        conversations_folder = self._folder_id(
            "conversations", self._folder_id(self.folder_name)
        )
        for file_id, name in self._file_ids(conversations_folder):
            if not name.endswith(".json"):
                continue
            conversation = self.read_json("conversations", name, {"messages": []})
            for message in conversation.get("messages", []):
                text = f"{message.get('role', 'unknown')}: {message.get('content', '')}"
                score = sum(term in text.lower() for term in terms)
                scored.append((score, text))
        return [text for score, text in sorted(scored, reverse=True)[:limit] if score > 0]
