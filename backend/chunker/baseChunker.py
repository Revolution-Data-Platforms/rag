


class baseChunker:
    def __init__(self):
        self._chunker = None
        self._chunkerType = None
        self._chunkerName = None
        
    def chunk(self, data):
        raise NotImplementedError("Chunking method not implemented")
    
    def chunkBatch(self, data):
        raise NotImplementedError("Batch chunking method not implemented")