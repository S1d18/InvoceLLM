"""
Work Scheduler для Invoice LLM.

Управление режимами работы для экономии энергии:
- NIGHT (23:00-06:00): Полная мощность LLM, batch processing
- DAY (06:00-23:00): Только cache, серверы в sleep
- FORCE: Ручной запуск LLM по требованию
"""

from __future__ import annotations

import logging
import socket
import struct
import subprocess
from datetime import datetime, time as dt_time
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class WorkMode(Enum):
    """Режимы работы."""
    NIGHT = "night"      # Полная мощность LLM
    DAY = "day"          # Только cache
    FORCE = "force"      # Ручной режим


class WorkScheduler:
    """
    Управление режимами работы для экономии энергии.

    Использование:
        scheduler = WorkScheduler(config)

        if scheduler.can_use_llm():
            result = llm.classify(text)
        else:
            result = cache.match(text)

        # Принудительный режим
        if scheduler.can_use_llm(force=True):
            scheduler.wake_servers()
            result = llm.classify(text)
    """

    def __init__(
        self,
        config: dict = None,
        night_start: str = "23:00",
        night_end: str = "06:00",
        servers: list[dict] = None,
        wake_on_lan: bool = True,
        timezone: str = None,
    ):
        """
        Инициализация scheduler.

        Args:
            config: Конфигурация (если задана, используется вместо параметров)
            night_start: Начало ночного режима (HH:MM)
            night_end: Конец ночного режима (HH:MM)
            servers: Список серверов
            wake_on_lan: Включить Wake-on-LAN
            timezone: Часовой пояс
        """
        if config:
            scheduler_config = config.get('scheduler', {})
            night_start = scheduler_config.get('night_start', night_start)
            night_end = scheduler_config.get('night_end', night_end)
            wake_on_lan = scheduler_config.get('wake_on_lan', wake_on_lan)
            timezone = scheduler_config.get('timezone', timezone)
            servers = config.get('servers', servers)

        self.night_start = self._parse_time(night_start)
        self.night_end = self._parse_time(night_end)
        self.servers = servers or []
        self.wol_enabled = wake_on_lan
        self.timezone = timezone

        self._force_mode = False

        logger.info(f"Scheduler initialized: NIGHT {night_start}-{night_end}, WoL: {wake_on_lan}")

    def _parse_time(self, time_str: str) -> dt_time:
        """Парсит строку времени HH:MM."""
        parts = time_str.split(':')
        return dt_time(int(parts[0]), int(parts[1]))

    @property
    def current_mode(self) -> WorkMode:
        """Текущий режим работы."""
        if self._force_mode:
            return WorkMode.FORCE

        now = datetime.now().time()

        # Ночной режим может пересекать полночь
        if self.night_start > self.night_end:
            # Например: 23:00 - 06:00
            if now >= self.night_start or now < self.night_end:
                return WorkMode.NIGHT
        else:
            # Например: 01:00 - 05:00
            if self.night_start <= now < self.night_end:
                return WorkMode.NIGHT

        return WorkMode.DAY

    def can_use_llm(self, force: bool = False) -> bool:
        """
        Можно ли использовать LLM сейчас.

        Args:
            force: Принудительно разрешить

        Returns:
            True если LLM доступен
        """
        if force:
            return True
        return self.current_mode in (WorkMode.NIGHT, WorkMode.FORCE)

    def enable_force_mode(self):
        """Включает принудительный режим."""
        self._force_mode = True
        logger.info("Force mode enabled")

    def disable_force_mode(self):
        """Выключает принудительный режим."""
        self._force_mode = False
        logger.info("Force mode disabled")

    def time_until_night(self) -> Optional[int]:
        """
        Возвращает время до ночного режима в секундах.

        Returns:
            Секунды до ночного режима или None если уже ночь
        """
        if self.current_mode == WorkMode.NIGHT:
            return None

        now = datetime.now()
        today = now.date()

        # Время начала ночи
        night_start_dt = datetime.combine(today, self.night_start)

        # Если время начала уже прошло, берём завтра
        if night_start_dt <= now:
            from datetime import timedelta
            night_start_dt += timedelta(days=1)

        return int((night_start_dt - now).total_seconds())

    def time_until_day(self) -> Optional[int]:
        """
        Возвращает время до дневного режима в секундах.

        Returns:
            Секунды до дневного режима или None если уже день
        """
        if self.current_mode == WorkMode.DAY:
            return None

        now = datetime.now()
        today = now.date()

        # Время начала дня
        day_start_dt = datetime.combine(today, self.night_end)

        # Если время начала уже прошло, берём завтра
        if day_start_dt <= now:
            from datetime import timedelta
            day_start_dt += timedelta(days=1)

        return int((day_start_dt - now).total_seconds())

    def wake_servers(self) -> list[str]:
        """
        Wake-on-LAN для серверов.

        Returns:
            Список разбуженных серверов
        """
        if not self.wol_enabled:
            logger.warning("Wake-on-LAN is disabled")
            return []

        woken = []

        for server in self.servers:
            mac = server.get('mac')
            name = server.get('name', server.get('host', 'unknown'))

            if mac:
                try:
                    self._send_magic_packet(mac)
                    woken.append(name)
                    logger.info(f"Sent WoL packet to {name} ({mac})")
                except Exception as e:
                    logger.error(f"Failed to wake {name}: {e}")

        return woken

    def sleep_servers(self) -> list[str]:
        """
        Отправляет серверы в sleep.

        Returns:
            Список серверов отправленных в sleep
        """
        slept = []

        for server in self.servers:
            ssh = server.get('ssh')
            name = server.get('name', server.get('host', 'unknown'))

            if ssh:
                try:
                    self._ssh_suspend(ssh)
                    slept.append(name)
                    logger.info(f"Sent sleep command to {name}")
                except Exception as e:
                    logger.error(f"Failed to sleep {name}: {e}")

        return slept

    def _send_magic_packet(self, mac: str):
        """
        Отправляет Magic Packet для Wake-on-LAN.

        Args:
            mac: MAC адрес в формате XX:XX:XX:XX:XX:XX
        """
        # Нормализуем MAC
        mac = mac.replace(':', '').replace('-', '').upper()

        if len(mac) != 12:
            raise ValueError(f"Invalid MAC address: {mac}")

        # Создаём magic packet
        # 6 байт 0xFF + 16 повторений MAC адреса
        mac_bytes = bytes.fromhex(mac)
        magic = b'\xff' * 6 + mac_bytes * 16

        # Отправляем broadcast на порт 9
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        try:
            sock.sendto(magic, ('255.255.255.255', 9))
        finally:
            sock.close()

    def _ssh_suspend(self, ssh_target: str):
        """
        Отправляет команду suspend через SSH.

        Args:
            ssh_target: SSH target в формате user@host
        """
        cmd = ['ssh', ssh_target, 'sudo', 'systemctl', 'suspend']

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=10,
            )
            if result.returncode != 0:
                logger.warning(f"SSH suspend returned {result.returncode}: {result.stderr.decode()}")
        except subprocess.TimeoutExpired:
            logger.warning(f"SSH suspend timeout for {ssh_target}")

    def get_status(self) -> dict:
        """Возвращает статус scheduler."""
        mode = self.current_mode

        status = {
            'mode': mode.value,
            'can_use_llm': self.can_use_llm(),
            'night_start': self.night_start.strftime('%H:%M'),
            'night_end': self.night_end.strftime('%H:%M'),
            'wol_enabled': self.wol_enabled,
            'force_mode': self._force_mode,
            'servers_count': len(self.servers),
        }

        if mode == WorkMode.DAY:
            seconds = self.time_until_night()
            if seconds:
                hours = seconds // 3600
                minutes = (seconds % 3600) // 60
                status['time_until_night'] = f"{hours}h {minutes}m"
        else:
            seconds = self.time_until_day()
            if seconds:
                hours = seconds // 3600
                minutes = (seconds % 3600) // 60
                status['time_until_day'] = f"{hours}h {minutes}m"

        return status


# Singleton instance
_scheduler: Optional[WorkScheduler] = None


def get_scheduler(config: dict = None) -> WorkScheduler:
    """Возвращает singleton экземпляр scheduler."""
    global _scheduler
    if _scheduler is None:
        _scheduler = WorkScheduler(config=config)
    return _scheduler
