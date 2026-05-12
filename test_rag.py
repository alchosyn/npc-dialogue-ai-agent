from src.npc_agent.tools import knowledge
import numpy as np

knowledge._ensure_loaded()
q = knowledge._model.encode("我妈接到电话说医保卡被人盗用了")
scores = np.dot(knowledge._doc_embeddings, q)
top = np.argsort(scores)[::-1][:3]
for i in top:
    print(round(float(scores[i]), 3), knowledge._knowledge_base[i]["title"])