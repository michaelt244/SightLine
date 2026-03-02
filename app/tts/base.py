from abc import ABC, abstractmethod


class TTSEngine(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def speak(self, text: str) -> None:
        raise NotImplementedError

    def interrupt(self) -> None:
        return

