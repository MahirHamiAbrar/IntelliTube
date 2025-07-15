from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSerializable


class SummarizerAgent:
    _map_chain: RunnableSerializable = None
    _reduce_chain: RunnableSerializable = None

    @property
    def map_chain(self) -> RunnableSerializable:
        if not self._map_chain:
            self._map_chain = None
        return self._map_chain
    
    @property
    def reduce_chain(self) -> RunnableSerializable:
        if not self._reduce_chain:
            self._reduce_chain = None
        return self._reduce_chain

    def __int__(self) -> None:
        pass