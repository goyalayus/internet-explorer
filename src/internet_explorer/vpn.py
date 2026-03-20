from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path

from pydantic import BaseModel, Field

from internet_explorer.config import AppConfig


class VpnStatus(BaseModel):
    running: bool
    pid: int | None = None
    started_by_this_call: bool = False
    ovpn_config: str | None = None
    pid_file: str
    log_file: str
    tunnel_interfaces: list[str] = Field(default_factory=list)
    default_route: str = ""
    docdb_host: str = ""
    docdb_port: int = 27017
    docdb_reachable: bool | None = None
    split_tunnel_ok: bool | None = None
    message: str = ""


class GenericVpnManager:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.log_dir = config.vpn_log_dir
        self.pid_file = self.log_dir / "openvpn.pid"
        self.log_file = self.log_dir / "openvpn.log"

    def status(self, *, check_docdb: bool = False) -> VpnStatus:
        pid = self._read_active_openvpn_pid()
        running = pid is not None
        tunnel_interfaces = self._list_tunnel_interfaces()
        default_route = self._default_route()
        docdb_reachable = self._check_docdb() if check_docdb and self.config.vpn_docdb_host else None
        split_tunnel_ok = None
        if default_route:
            split_tunnel_ok = re.search(r" dev tun[0-9]+", default_route) is None
        return VpnStatus(
            running=running,
            pid=pid if running else None,
            ovpn_config=str(self.config.vpn_ovpn_config) if self.config.vpn_ovpn_config else None,
            pid_file=str(self.pid_file),
            log_file=str(self.log_file),
            tunnel_interfaces=tunnel_interfaces,
            default_route=default_route,
            docdb_host=self.config.vpn_docdb_host,
            docdb_port=self.config.vpn_docdb_port,
            docdb_reachable=docdb_reachable,
            split_tunnel_ok=split_tunnel_ok,
            message="running" if running else "stopped",
        )

    def ensure_started(self) -> VpnStatus:
        current = self.status(check_docdb=False)
        if current.running:
            current.message = "already running"
            return current
        return self.start()

    def start(self) -> VpnStatus:
        self._ensure_start_prerequisites()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._remove_stale_pid_file()
        if self.log_file.exists():
            self.log_file.unlink()

        route_before = self._default_route()
        if self.config.vpn_start_script and self.config.vpn_start_script.exists():
            command = [
                "bash",
                str(self.config.vpn_start_script),
                "--config",
                str(self._require_ovpn_config()),
                "--pid-file",
                str(self.pid_file),
                "--log-file",
                str(self.log_file),
            ]
        else:
            command = [
                "sudo",
                "openvpn",
                "--config",
                str(self._require_ovpn_config()),
                "--daemon",
                "--writepid",
                str(self.pid_file),
                "--log",
                str(self.log_file),
            ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        self._wait_for_pid_and_tunnel()

        route_after = self._default_route()
        if self.config.vpn_require_split_tunnel and re.search(r" dev tun[0-9]+", route_after):
            self.stop()
            raise RuntimeError(
                "VPN changed the default route to a tunnel interface. "
                "Set VPN_REQUIRE_SPLIT_TUNNEL=false if you intentionally want full-tunnel."
            )

        docdb_reachable = None
        if self.config.vpn_docdb_host:
            docdb_reachable = self._check_docdb()
            if self.config.vpn_require_docdb_reachable and not docdb_reachable:
                self.stop()
                raise RuntimeError(
                    f"VPN started but DocDB {self.config.vpn_docdb_host}:{self.config.vpn_docdb_port} was not reachable."
                )

        status = self.status(check_docdb=False)
        status.started_by_this_call = True
        status.docdb_reachable = docdb_reachable
        status.split_tunnel_ok = re.search(r" dev tun[0-9]+", route_after) is None
        status.default_route = route_after
        status.message = f"started (default route before: {route_before or 'unknown'})"
        return status

    def stop(self) -> VpnStatus:
        self._ensure_base_dependencies()
        pid = self._read_active_openvpn_pid()
        if pid is None:
            self._remove_stale_pid_file()
            status = self.status(check_docdb=False)
            status.message = "already stopped"
            return status

        subprocess.run(["sudo", "kill", str(pid)], check=True, capture_output=True, text=True)
        for _ in range(20):
            if not self._process_exists(pid):
                break
            time.sleep(0.5)
        self._remove_stale_pid_file()
        status = self.status(check_docdb=False)
        status.message = "stopped"
        return status

    def _require_ovpn_config(self) -> Path:
        if self.config.vpn_ovpn_config is None or not self.config.vpn_ovpn_config.exists():
            raise RuntimeError(
                "No OVPN config found. Set VPN_OVPN_CONFIG or place client-config-staging.ovpn "
                "in vpn/, the repo root, or a sibling query optimizer checkout."
            )
        return self.config.vpn_ovpn_config

    def _ensure_base_dependencies(self) -> None:
        for dependency in ("sudo", "ip"):
            if shutil.which(dependency) is None:
                raise RuntimeError(f"Missing required dependency: {dependency}")

    def _ensure_start_prerequisites(self) -> None:
        self._ensure_base_dependencies()
        for dependency in ("openvpn",):
            if shutil.which(dependency) is None:
                raise RuntimeError(f"Missing required dependency: {dependency}")
        sudo_check = subprocess.run(["sudo", "-n", "true"], capture_output=True, text=True)
        if sudo_check.returncode != 0:
            raise RuntimeError("sudo requires a password or is unavailable for non-interactive use.")

    def _wait_for_pid_and_tunnel(self) -> None:
        deadline = time.time() + 30
        while time.time() < deadline:
            pid = self._read_active_openvpn_pid()
            if pid and self._list_tunnel_interfaces():
                return
            time.sleep(1)
        log_tail = self._tail_log()
        raise RuntimeError(f"OpenVPN did not come up within 30s. Log tail:\n{log_tail}")

    def _tail_log(self, max_lines: int = 40) -> str:
        if not self.log_file.exists():
            return "<no log file>"
        lines = self.log_file.read_text(errors="ignore").splitlines()
        return "\n".join(lines[-max_lines:])

    def _read_pid(self) -> int | None:
        if not self.pid_file.exists():
            return None
        raw = self.pid_file.read_text().strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    def _remove_stale_pid_file(self) -> None:
        pid = self._read_pid()
        if pid is None:
            self.pid_file.unlink(missing_ok=True)
            return

        if not self._process_is_openvpn(pid):
            self.pid_file.unlink(missing_ok=True)

    def _read_active_openvpn_pid(self) -> int | None:
        pid = self._read_pid()
        if pid is None:
            return None

        if self._process_is_openvpn(pid):
            return pid

        self.pid_file.unlink(missing_ok=True)
        return None

    def _process_exists(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    def _process_is_openvpn(self, pid: int) -> bool:
        if not self._process_exists(pid):
            return False

        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "comm="],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return False

        process_name = result.stdout.strip().lower()
        if not process_name:
            return False

        return "openvpn" in process_name

    def _list_tunnel_interfaces(self) -> list[str]:
        result = subprocess.run(["ip", "-o", "link", "show"], capture_output=True, text=True, check=True)
        interfaces = []
        for line in result.stdout.splitlines():
            parts = line.split(":")
            if len(parts) < 2:
                continue
            name = parts[1].strip()
            if name.startswith("tun"):
                interfaces.append(name)
        return interfaces

    def _default_route(self) -> str:
        result = subprocess.run(["ip", "route", "show", "default"], capture_output=True, text=True, check=True)
        return result.stdout.splitlines()[0].strip() if result.stdout.strip() else ""

    def _check_docdb(self) -> bool:
        try:
            with socket.create_connection((self.config.vpn_docdb_host, self.config.vpn_docdb_port), timeout=3):
                return True
        except OSError:
            return False
