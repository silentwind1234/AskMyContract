import faiss
from faiss import write_index, read_index
class FaissIdx:

    def __init__(self, dim=1536):
        self.index = faiss.IndexFlatIP(dim)
        self.doc_map = dict()
        self.ctr = 0

    def add_doc(self, document_text,vector):
        self.index.add(vector)
        self.doc_map[self.ctr] = document_text
        self.ctr += 1

    def search_doc(self, query, top=5):
        D, I = self.index.search(self.model.get_embedding(query), top)
        print(I)
        return "\n\n".join([self.doc_map[idx] for idx in I if idx in self.doc_map])
    
    def save(self,name):
        write_index(self.index,name)

    def load(self,name):
        self.index = read_index(name)