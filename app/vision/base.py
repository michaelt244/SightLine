from abc import ABC, abstractmethod


class VisionEngine(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def describe(self, frame_b64: str, prompt: str) -> str:
        raise NotImplementedError

